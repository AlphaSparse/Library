#include "alphasparse_spmm_utils.h"
#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include <iostream>


template<int block_size,
         typename T>
static __global__ void
__launch_bounds__(512)
adaptive_partition_kernel(
  T M, T nnz, T items_per_seg,
  const T* __restrict__ csr_row_ptr,
  T* seg_row_id)
{
  __shared__ T csr_seg_row_ptr[block_size + 1];

  T local_tid = threadIdx.x;
  T blockid = blockIdx.x;
  T tid = blockid * blockDim.x + local_tid;
  T stride = CEIL(M, block_size);

  T csr_row_ptr_idx = MIN(((local_tid + 1) * stride), M);
  if(local_tid == 0) {
    csr_seg_row_ptr[0] = 0;
  }
  csr_seg_row_ptr[local_tid + 1] = __ldg(&csr_row_ptr[csr_row_ptr_idx]);

  __syncthreads();

  T seg_eleid = tid * items_per_seg;

  if (seg_eleid < nnz) {
    T csr_seg_row_ptr_id = binary_search_segment_number<T>(csr_seg_row_ptr, block_size, seg_eleid);

    T low = csr_seg_row_ptr_id * stride;
    T hi = MIN(low + stride, M);
    seg_row_id[tid] = low + binary_search_segment_number<T>(csr_row_ptr + low, hi - low, seg_eleid);

    if (seg_eleid + items_per_seg >= nnz) {
      seg_row_id[tid + 1] = M;
    }
  }
}

template<int block_size,
         int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_adaptive_aligned_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_row_id)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];
  __shared__ T s_seg_row_id[2];

  T blockid = blockIdx.x;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_offset = threadIdx.y * blockDim.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);
  U col_val[factor] = {};
  W sum[factor] = {};


  if (local_tid < 2) {
    s_seg_row_id[local_tid] = seg_row_id[blockid + local_tid];
  }

  __syncthreads();

  T seg_row_start = s_seg_row_id[0];
  T seg_row_end = s_seg_row_id[1];
  T seg_row_num = seg_row_end - seg_row_start;
  T row_offset = 0;

  if (seg_row_num > 0) {
    // check the first row
    T row_start = csr_row_ptr[seg_row_start];
    T row_end = csr_row_ptr[seg_row_start + 1];
    T row_length = row_end - row_start;
    T work_size = CEIL(nnz, M);
    // very long row.
    if (row_length > work_size * block_size / warp_size) {

      T rowid = seg_row_start;
      if (local_warpid == 0) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (ORDER_ROW) {
            matC[rowid * ldc + colid] *= beta;
          } else {
            matC[colid * ldc + rowid] *= beta;
          }
        }
      }

      __syncthreads();

      T stride = CEIL(row_length, block_size / warp_size);
      T nnz_offset = local_warpid * stride + row_start;

      for (T i = 0; i < stride; i += warp_size) {
        if (i + warp_laneid < stride && i + warp_laneid < row_end) {
          s_val[local_tid] = csr_val[nnz_offset + i + warp_laneid];
          s_col_ind[local_tid] = csr_col_ind[nnz_offset + i + warp_laneid];
        }
        for (T j = 0; j < warp_size; j++) {
          if (i + j >= stride || i + j + nnz_offset >= row_end) break;
          T rowid_B = s_col_ind[local_offset + j];

          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (ORDER_ROW) {
              sum[k] += s_val[local_offset + j] * matB[rowid_B * ldb + colid];
            } else {
              sum[k] += s_val[local_offset + j] * matB[colid * ldb + rowid_B];
            }
          }
        }
      }
      if (warp_size < PHYSICAL_WARP_SIZE) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = warpReduce<W, PHYSICAL_WARP_SIZE, warp_size>(sum[k]);
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if ((local_warpid & (PHYSICAL_WARP_SIZE / warp_size - 1)) == 0) {
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
            }
          }
        }
      } else {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (ORDER_ROW) {
            atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
          } else {
            atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
          }
        }
      }
      row_offset = 1;
    }
    seg_row_start += row_offset;
    seg_row_num -= row_offset;
    //can optimize.
    for (T i = 0; i < seg_row_num; i += block_size / warp_size) {
      if (i + local_warpid < seg_row_num) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
            sum[k] = {};
        }
        T rowid = seg_row_start + i + local_warpid;
        T row_start = csr_row_ptr[rowid];
        T row_end = csr_row_ptr[rowid + 1];
        T eleid = row_start;
        for (T i = row_start; i < row_end; i += warp_size) {
          eleid += warp_laneid;
          if (eleid < row_end) {
            s_val[local_tid] = csr_val[eleid];
            s_col_ind[local_tid] = csr_col_ind[eleid];
          }

          eleid -= warp_laneid;

          for (T j = 0; j < warp_size; j++) {
            if (eleid >= row_end) break;
            T rowid_B = s_col_ind[local_offset + j];

            #pragma unroll
            for (int k = 0; k < factor; k++) {
              T colid = block_colid + k * warp_size + warp_laneid; 
              if (ORDER_ROW) {
                sum[k] += s_val[local_offset + j] * matB[rowid_B * ldb + colid];
              } else {
                sum[k] += s_val[local_offset + j] * matB[colid * ldb + rowid_B];
              }
            }
            eleid++;
          }
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (ORDER_ROW) {
            matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
          } else {
            matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
          }
        }
      }
    }
  }
}

template<int block_size,
         int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_adaptive_nonaligned_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_row_id)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];
  __shared__ T s_seg_row_id[2];

  T blockid = blockIdx.x;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_offset = threadIdx.y * blockDim.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);
  U col_val[factor] = {};
  W sum[factor] = {};


  if (local_tid < 2) {
    s_seg_row_id[local_tid] = seg_row_id[blockid + local_tid];
  }

  __syncthreads();

  T seg_row_start = s_seg_row_id[0];
  T seg_row_end = s_seg_row_id[1];
  T seg_row_num = seg_row_end - seg_row_start;
  T row_offset = 0;

  if (seg_row_num > 0) {
    // check the first row
    T row_start = csr_row_ptr[seg_row_start];
    T row_end = csr_row_ptr[seg_row_start + 1];
    T row_length = row_end - row_start;
    T work_size = CEIL(nnz, M);
    // very long row.
    if (row_length > work_size * block_size / warp_size) {

      T rowid = seg_row_start;
      if (local_warpid == 0) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              matC[rowid * ldc + colid] *= beta;
            } else {
              matC[colid * ldc + rowid] *= beta;
            }
          }
        }
      }

      __syncthreads();

      T stride = CEIL(row_length, block_size / warp_size);
      T nnz_offset = local_warpid * stride + row_start;

      for (T i = 0; i < stride; i += warp_size) {
        if (i + warp_laneid < stride && i + warp_laneid < row_end) {
          s_val[local_tid] = csr_val[nnz_offset + i + warp_laneid];
          s_col_ind[local_tid] = csr_col_ind[nnz_offset + i + warp_laneid];
        }
        for (T j = 0; j < warp_size; j++) {
          if (i + j >= stride || i + j + nnz_offset >= row_end) break;
          T rowid_B = s_col_ind[local_offset + j];

          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                sum[k] += s_val[local_offset + j] * matB[rowid_B * ldb + colid];
              } else {
                sum[k] += s_val[local_offset + j] * matB[colid * ldb + rowid_B];
              }
            }
          }
        }
      }
      if (warp_size < PHYSICAL_WARP_SIZE) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = warpReduce<W, PHYSICAL_WARP_SIZE, warp_size>(sum[k]);
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if ((local_warpid & (PHYSICAL_WARP_SIZE / warp_size - 1)) == 0) {
              if (ORDER_ROW) {
                atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
              } else {
                atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
              }
            }
          }
        }
      } else {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
            }
          }
        }
      }
      row_offset = 1;
    }
    seg_row_start += row_offset;
    seg_row_num -= row_offset;
    //can optimize.
    for (T i = 0; i < seg_row_num; i += block_size / warp_size) {
      if (i + local_warpid < seg_row_num) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
            sum[k] = {};
        }
        T rowid = seg_row_start + i + local_warpid;
        T row_start = csr_row_ptr[rowid];
        T row_end = csr_row_ptr[rowid + 1];
        T eleid = row_start;
        for (T i = row_start; i < row_end; i += warp_size) {
          eleid += warp_laneid;
          if (eleid < row_end) {
            s_val[local_tid] = csr_val[eleid];
            s_col_ind[local_tid] = csr_col_ind[eleid];
          }

          eleid -= warp_laneid;

          for (T j = 0; j < warp_size; j++) {
            if (eleid >= row_end) break;
            T rowid_B = s_col_ind[local_offset + j];

            #pragma unroll
            for (int k = 0; k < factor; k++) {
              T colid = block_colid + k * warp_size + warp_laneid; 
              if (colid < N) {
                if (ORDER_ROW) {
                  sum[k] += s_val[local_offset + j] * matB[rowid_B * ldb + colid];
                } else {
                  sum[k] += s_val[local_offset + j] * matB[colid * ldb + rowid_B];
                }
              }
            }
            eleid++;
          }
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
            } else {
              matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
            }
          }
        }
      }
    }
  }
}
template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_adaptive(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csr_row_ptr,
                const T* __restrict__ csr_col_ind,
                const U* __restrict__ csr_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  const int block_size = RefThreadPerBlock;
  int warp_size, factor;
  if (ORDER_ROW) {
    if (N > 128) {
      warp_size = 64;
      factor = 4;
    }
    else if (N > 64) {
      warp_size = 32;
      factor = 4;
    }
    else if (N > 32) {
      warp_size = 16;
      factor = 4;
    }
    else if (N > 16) {
      warp_size = 16;
      factor = 2;
    }
    else if (N > 8) {
      warp_size = 16;
      factor = 1;
    }
    else if (N > 4) {
      warp_size = 8;
      factor = 1;
    }
    else {
      warp_size = 4;
      factor = 1;
    }
  }
  else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  }

  bool align = !(N % (warp_size * factor));

  const T work_size = CEIL(nnz, M);
  const T block_work_size = work_size * block_size / warp_size;
  T Mdim_worker = CEIL(nnz, work_size);
  T block_num = CEIL(nnz, block_work_size);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(block_num, 2 * RefThreadPerBlock);
  adaptive_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    M, nnz, block_work_size, csr_row_ptr, seg_row_id);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Mdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);

  if (align) {
    if (warp_size == 64 && factor == 4) {
      csrspmm_adaptive_aligned_kernel<block_size, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_adaptive_aligned_kernel<block_size, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_adaptive_aligned_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_adaptive_aligned_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 4) {
      csrspmm_adaptive_aligned_kernel<block_size, 16, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_adaptive_aligned_kernel<block_size, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_adaptive_aligned_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_adaptive_aligned_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_adaptive_aligned_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
  }
  else {
    if (warp_size == 64 && factor == 4) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 4) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 16, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_adaptive_nonaligned_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }

  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}