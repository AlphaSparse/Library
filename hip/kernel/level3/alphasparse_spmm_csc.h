#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include <iostream>

#define FULL_MASK 0xffffffff
#define RefThreadPerBlock 256
#define PHYSICAL_WARP_SIZE 64

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

// assigns a thread to an output element.
template<int block_size,
         int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
cscspmm_colbalance_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csc_col_ptr,
  const T* __restrict__ csc_row_ind,
  const U* __restrict__ csc_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc)
{
  __shared__ U s_val[block_size];
  __shared__ T s_row_ind[block_size];

  T colid_A = blockIdx.x * blockDim.y + threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;
  T local_offset = threadIdx.y * blockDim.x;
  T local_tid = local_offset + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  U col_val[factor] = {};
  if (colid_A < K) {
    T colstart = csc_col_ptr[colid_A];
    T colend = csc_col_ptr[colid_A + 1];
    T eleid = colstart;

    T rowid_B = colid_A;
    // load col_val from matB
    #pragma unroll
    for (int k = 0; k < factor; k++) {
      T colid = block_colid + k * warp_size + warp_laneid; 
      if (colid < N) {
        if (ORDER_ROW) {
          col_val[k] = matB[rowid_B * ldb + colid];
        } else {
          col_val[k] = matB[colid * ldb + rowid_B];
        }
      }
    }

    // start compute
    for (T i = colstart; i < colend; i += warp_size) {
      eleid += warp_laneid;
      if (eleid < colend) {
        s_val[local_tid] = csc_val[eleid];
        s_row_ind[local_tid] = csc_row_ind[eleid];
      }

      eleid -= warp_laneid;

      for (T j = 0; j < warp_size; j++) {
        if (eleid >= colend) break;
        T rowid = s_row_ind[local_offset + j];

        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * s_val[local_offset + j] * col_val[k]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * s_val[local_offset + j] * col_val[k]);
            }
          }
        }
        eleid++;
      }
    }
  }
}

template<int block_size,
         int warp_size,
         int work_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
cscspmm_elebalance_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csc_col_ptr,
  const T* __restrict__ csc_row_ind,
  const U* __restrict__ csc_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_col_id)
{
  __shared__ U s_val[block_size / warp_size * work_size];
  __shared__ T s_row_ind[block_size / warp_size * work_size];
  __shared__ T s_seg_col_id[block_size / warp_size + 1];

  T block_segid = blockIdx.x * blockDim.y;
  T segid = block_segid + threadIdx.y;
  T element_per_seg = work_size;
  T eleid = segid * element_per_seg;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T local_segid = local_warpid;
  T block_eleid = blockIdx.x * blockDim.y * element_per_seg;
  T warp_laneid = local_tid & (warp_size - 1);
  U col_val[factor] = {};


  if ((local_tid < block_size / warp_size + 1) && (block_eleid + local_tid * work_size < nnz + work_size)) {
    s_seg_col_id[local_tid] = seg_col_id[block_segid + local_tid];
  }

  __syncthreads();

  if (warp_laneid < work_size && eleid + warp_laneid < nnz) {
    s_val[local_warpid * work_size + warp_laneid] = csc_val[eleid + warp_laneid];
    s_row_ind[local_warpid * work_size + warp_laneid] = csc_row_ind[eleid + warp_laneid];
  }

  if (eleid < nnz) {
    T seg_col_start = s_seg_col_id[local_segid];
    T seg_col_end = s_seg_col_id[local_segid + 1];
    T seg_col_length = seg_col_end - seg_col_start;
    T rowid_B = seg_col_start;
    #pragma unroll
    for (int k = 0; k < factor; k++) {
      T colid = block_colid + k * warp_size + warp_laneid; 
      if (colid < N) {
        if (ORDER_ROW) {
          col_val[k] = matB[rowid_B * ldb + colid];
        } else {
          col_val[k] = matB[colid * ldb + rowid_B];
        }
      }
    }
    T step = csc_col_ptr[rowid_B + 1] - eleid; //remaining elements in this row.
    for (T i = 0; i < element_per_seg; i++) {
      if (eleid >= nnz) break;
      T s_eleid = eleid - block_eleid;

      T rowid = s_row_ind[s_eleid];
      if (i < step) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * s_val[s_eleid] * col_val[k]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * s_val[s_eleid] * col_val[k]);
            }
          }
        }
        eleid++;
      } else {
        // next element may cross multicols.
        T seg_colid = binary_search_segment_number<T>(csc_col_ptr + seg_col_start, seg_col_length, eleid);
        rowid_B = seg_col_start + seg_colid;
        step += csc_col_ptr[rowid_B + 1] - eleid;
        if (factor > 1) {
          // load new col_val from matB
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                col_val[k] = matB[rowid_B * ldb + colid];
              } else {
                col_val[k] = matB[colid * ldb + rowid_B];
              }
            }
          }
          // compute
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                atomicAdd(&matC[rowid * ldc + colid], alpha * s_val[s_eleid] * col_val[k]);
              } else {
                atomicAdd(&matC[colid * ldc + rowid], alpha * s_val[s_eleid] * col_val[k]);
              }
            }
          }
        } else {
          T colid = block_colid + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              col_val[0] = matB[rowid_B * ldb + colid];
              atomicAdd(&matC[rowid * ldc + colid], alpha * s_val[s_eleid] * col_val[0]);
            } else {
              col_val[0] = matB[colid * ldb + rowid_B];
              atomicAdd(&matC[colid * ldc + rowid], alpha * s_val[s_eleid] * col_val[0]);
            }
          }
        }
        eleid++;
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
cscspmm_adaptive_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csc_col_ptr,
  const T* __restrict__ csc_row_ind,
  const U* __restrict__ csc_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ seg_col_id)
{
  __shared__ U s_val[block_size];
  __shared__ T s_row_ind[block_size];
  __shared__ T s_seg_col_id[2];

  T blockid = blockIdx.x;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_offset = threadIdx.y * blockDim.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);
  U col_val[factor] = {};


  if (local_tid < 2) {
    s_seg_col_id[local_tid] = seg_col_id[blockid + local_tid];
  }

  __syncthreads();

  T seg_col_start = s_seg_col_id[0];
  T seg_col_end = s_seg_col_id[1];
  T seg_col_num = seg_col_end - seg_col_start;
  T col_offset = 0;

  if (seg_col_num > 0) {
    // check the first col
    T col_start = csc_col_ptr[seg_col_start];
    T col_end = csc_col_ptr[seg_col_start + 1];
    T col_length = col_end - col_start;
    T work_size = CEIL(nnz, K);
    // very long col.
    if (col_length > work_size * block_size / warp_size) {

      T rowid_B = seg_col_start;
      // load col_val from matB.
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        if (colid < N) {
          if (ORDER_ROW) {
            col_val[k] = matB[rowid_B * ldb + colid];
          } else {
            col_val[k] = matB[colid * ldb + rowid_B];
          }
        }
      }

      T stride = CEIL(col_length, block_size / warp_size);
      T nnz_offset = local_warpid * stride + col_start;

      for (T i = 0; i < stride; i += warp_size) {
        if (i + warp_laneid < stride && i + warp_laneid < col_end) {
          s_val[local_tid] = csc_val[nnz_offset + i + warp_laneid];
          s_row_ind[local_tid] = csc_row_ind[nnz_offset + i + warp_laneid];
        }
        for (T j = 0; j < warp_size; j++) {
          if (i + j >= stride || i + j + nnz_offset >= col_end) break;
          T rowid = s_row_ind[local_offset + j];

          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (colid < N) {
              if (ORDER_ROW) {
                atomicAdd(&matC[rowid * ldc + colid], alpha * s_val[local_offset + j] * col_val[k]);
              } else {
                atomicAdd(&matC[colid * ldc + rowid], alpha * s_val[local_offset + j] * col_val[k]);
              }
            }
          }
        }
      }
      col_offset = 1;
    }
    seg_col_start += col_offset;
    seg_col_num -= col_offset;
    //can optimize.
    for (T i = 0; i < seg_col_num; i += block_size / warp_size) {
      if (i + local_warpid < seg_col_num) {
        T rowid_B = seg_col_start + i + local_warpid;
        T col_start = csc_col_ptr[rowid_B];
        T col_end = csc_col_ptr[rowid_B + 1];
        T eleid = col_start;

        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              col_val[k] = matB[rowid_B * ldb + colid];
            } else {
              col_val[k] = matB[colid * ldb + rowid_B];
            }
          }
        }

        for (T i = col_start; i < col_end; i += warp_size) {
          eleid += warp_laneid;
          if (eleid < col_end) {
            s_val[local_tid] = csc_val[eleid];
            s_row_ind[local_tid] = csc_row_ind[eleid];
          }

          eleid -= warp_laneid;

          for (T j = 0; j < warp_size; j++) {
            if (eleid >= col_end) break;
            T rowid = s_row_ind[local_offset + j];


            #pragma unroll
            for (int k = 0; k < factor; k++) {
              T colid = block_colid + k * warp_size + warp_laneid; 
              if (colid < N) {
                if (ORDER_ROW) {
                  atomicAdd(&matC[rowid * ldc + colid], alpha * s_val[local_offset + j] * col_val[k]);
                } else {
                  atomicAdd(&matC[colid * ldc + rowid], alpha * s_val[local_offset + j] * col_val[k]);
                }
              }
            }
            eleid++;
          }
        }
      }
    }
  }
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
cscspmm_cb(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csc_col_ptr,
                const T* __restrict__ csc_row_ind,
                const U* __restrict__ csc_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  int warp_size, factor;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  }
  const int block_size = RefThreadPerBlock;

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T Kdim_worker = K;
  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Kdim_thread_per_tb = CEIL(block_size, Ndim_thread_per_tb);
  T Kdim_threadblock = CEIL(Kdim_worker, Kdim_thread_per_tb);

  dim3 gridDim(Kdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Kdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    cscspmm_colbalance_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 32 && factor == 2) {
    cscspmm_colbalance_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 32 && factor == 1) {
    cscspmm_colbalance_kernel<block_size, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 16 && factor == 1) {
    cscspmm_colbalance_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 8 && factor == 1) {
    cscspmm_colbalance_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc);
  }
  if (warp_size == 4 && factor == 1) {
    cscspmm_colbalance_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
cscspmm_eb(alphasparseHandle_t handle,
            T M, T N, T K, T nnz, W alpha, 
            const T* __restrict__ csc_col_ptr,
            const T* __restrict__ csc_row_ind,
            const U* __restrict__ csc_val,
            const U* __restrict__ matB,
            T ldb,  W beta,
            V* __restrict__ matC,
            T ldc,
            void* externalBuffer)
{
  int warp_size, factor, work_size;
  const int block_size = RefThreadPerBlock;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
      work_size = 32;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
      work_size = 32;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
      work_size = 32;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
      work_size = 16;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
      work_size = 8;
    } else {
      warp_size = 4;
      factor = 1;
      work_size = 4;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
      work_size = 8;
    } else {
      warp_size = 4;
      factor = 1;
      work_size = 4;
    }
  }

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T Nnzdim_worker = CEIL(nnz, (T)work_size);

  T *seg_col_id;
  seg_col_id = reinterpret_cast<T *>(externalBuffer);
  blocksPerGrid = CEIL(Nnzdim_worker, 2 * RefThreadPerBlock);
  elebalance_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    K, nnz, work_size, csc_col_ptr, seg_col_id);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    cscspmm_elebalance_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 32 && factor == 2) {
    cscspmm_elebalance_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 32 && factor == 1) {
    cscspmm_elebalance_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 16 && factor == 1) {
    cscspmm_elebalance_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 8 && factor == 1) {
    cscspmm_elebalance_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 4 && factor == 1) {
    cscspmm_elebalance_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
cscspmm_adaptive(alphasparseHandle_t handle,
                T M, T N, T K, T nnz, W alpha, 
                const T* __restrict__ csc_col_ptr,
                const T* __restrict__ csc_row_ind,
                const U* __restrict__ csc_val,
                const U* __restrict__ matB,
                T ldb,  W beta,
                V* __restrict__ matC,
                T ldc,
                void* externalBuffer)
{
  const int block_size = RefThreadPerBlock;
  int warp_size, factor;
  if (ORDER_ROW) {
    if (N > 64) {
      warp_size = 32;
      factor = 4;
    } else if (N > 32) {
      warp_size = 32;
      factor = 2;
    } else if (N > 16) {
      warp_size = 32;
      factor = 1;
    } else if (N > 8) {
      warp_size = 16;
      factor = 1;
    } else if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  } else {
    if (N > 4) {
      warp_size = 8;
      factor = 1;
    } else {
      warp_size = 4;
      factor = 1;
    }
  }

  const T work_size = CEIL(nnz, K);
  const T block_work_size = work_size * block_size / warp_size;
  T Kdim_worker = CEIL(nnz, work_size);
  T block_num = CEIL(nnz, block_work_size);

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T *seg_col_id;
  seg_col_id = reinterpret_cast<T *>(externalBuffer);
  threadsPerBlock = RefThreadPerBlock;
  blocksPerGrid = CEIL(block_num, 2 * RefThreadPerBlock);
  adaptive_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    K, nnz, block_work_size, csc_col_ptr, seg_col_id);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Kdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Kdim_threadblock = CEIL(Kdim_worker, Kdim_thread_per_tb);

  dim3 gridDim(Kdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Kdim_thread_per_tb, 1);
  if (warp_size == 32 && factor == 4) {
    cscspmm_adaptive_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 32 && factor == 2) {
    cscspmm_adaptive_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 32 && factor == 1) {
    cscspmm_adaptive_kernel<block_size, 32, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 16 && factor == 1) {
    cscspmm_adaptive_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 8 && factor == 1) {
    cscspmm_adaptive_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  if (warp_size == 4 && factor == 1) {
    cscspmm_adaptive_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csc_col_ptr, csc_row_ind, csc_val, matB, ldb,
        beta, matC, ldc,
        seg_col_id);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}