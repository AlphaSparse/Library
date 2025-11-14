#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include "alphasparse_spmm_utils.h"
#include <iostream>
#include <algorithm>

template<int block_size,
         typename T>
static __global__ void
__launch_bounds__(512)
elebalance_partition_kernel(
  T M, T nnz, T items_per_seg,
  const T* __restrict__ csr_row_ptr,
  T* seg_row_id)
{
  __shared__ T csr_seg_row_ptr[block_size + 1];

  T local_tid = threadIdx.x;
  T blockid = blockIdx.x;
  T tid = blockid * blockDim.x + local_tid;
  T stride = CEIL(M, block_size);

  T block_eleid_start = blockid * block_size * items_per_seg;
  T block_eleid_end = MIN((blockid + 1)* block_size * items_per_seg, nnz);


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
         int work_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_nnz_split_aligned_kernel(
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
  __shared__ U s_val[block_size / warp_size * work_size];
  __shared__ T s_col_ind[block_size / warp_size * work_size];
  __shared__ T s_seg_row_id[block_size / warp_size + 1];

  T block_segid = blockIdx.x * blockDim.y;
  T segid = block_segid + threadIdx.y;
  T element_per_seg = work_size;
  T eleid = segid * element_per_seg;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T local_segid = local_warpid;
  T block_eleid = block_segid * element_per_seg;
  T warp_laneid = local_tid & (warp_size - 1);
  W sum[factor] = {};
  W zero = {};


  if ((local_tid < block_size / warp_size + 1) && (block_eleid + local_tid * work_size < nnz + work_size)) {
    s_seg_row_id[local_tid] = seg_row_id[block_segid + local_tid];
  }

  __syncthreads();
   
  if (eleid >= nnz) return;

  if (warp_laneid < work_size && eleid + warp_laneid < nnz) {
    s_val[local_warpid * work_size + warp_laneid] = csr_val[eleid + warp_laneid];
    s_col_ind[local_warpid * work_size + warp_laneid] = csr_col_ind[eleid + warp_laneid];
  }

  T seg_row_start = s_seg_row_id[local_segid];
  T seg_row_length = s_seg_row_id[local_segid + 1] - seg_row_start;
  T rowid = seg_row_start;
  T step = csr_row_ptr[rowid + 1] - eleid; //remaining elements in this row.
  for (T i = 0; i < element_per_seg; i++) {
    if (eleid >= nnz) break;
    T s_eleid =  eleid - block_eleid;

    T rowid_B = s_col_ind[s_eleid];
    U val = s_val[s_eleid];

    if (i < step) {
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        if (ORDER_ROW) {
          sum[k] += val * matB[rowid_B * ldb + colid];
        } else {
          sum[k] += val * matB[colid * ldb + rowid_B];
        }
      }
      eleid++;
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
      // next element may cross multirows.
      T seg_rowid = binary_search_segment_number<T>(csr_row_ptr + seg_row_start, seg_row_length, eleid);
      rowid = seg_row_start + seg_rowid;
      step += csr_row_ptr[rowid + 1] - eleid;
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        sum[k] = zero;
        if (ORDER_ROW) {
          sum[k] += val * matB[rowid_B * ldb + colid];
        } else {
          sum[k] += val * matB[colid * ldb + rowid_B];
        }
      }
      eleid++;
    }
  }
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
csrspmm_nnz_split_nonaligned_kernel(
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
  __shared__ U s_val[block_size / warp_size * work_size];
  __shared__ T s_col_ind[block_size / warp_size * work_size];
  __shared__ T s_seg_row_id[block_size / warp_size + 1];

  T block_segid = blockIdx.x * blockDim.y;
  T segid = block_segid + threadIdx.y;
  T element_per_seg = work_size;
  T eleid = segid * element_per_seg;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T local_segid = local_warpid;
  T block_eleid = block_segid * element_per_seg;
  T warp_laneid = local_tid & (warp_size - 1);
  W sum[factor] = {};
  W zero = {};


  if ((local_tid < block_size / warp_size + 1) && (block_eleid + local_tid * work_size < nnz + work_size)) {
    s_seg_row_id[local_tid] = seg_row_id[block_segid + local_tid];
  }

  __syncthreads();

  if (warp_laneid < work_size && eleid + warp_laneid < nnz) {
    s_val[local_warpid * work_size + warp_laneid] = csr_val[eleid + warp_laneid];
    s_col_ind[local_warpid * work_size + warp_laneid] = csr_col_ind[eleid + warp_laneid];
  }

  if (eleid < nnz) {
    T seg_row_start = s_seg_row_id[local_segid];
    T seg_row_length = s_seg_row_id[local_segid + 1] - seg_row_start;
    T rowid = seg_row_start;
    T step = csr_row_ptr[rowid + 1] - eleid; //remaining elements in this row.
    for (T i = 0; i < element_per_seg; i++) {
      if (eleid >= nnz) break;
      T s_eleid =  eleid - block_eleid;

      T rowid_B = s_col_ind[s_eleid];
      U val = s_val[s_eleid];

      if (i < step) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              sum[k] += val * matB[rowid_B * ldb + colid];
            } else {
              sum[k] += val * matB[colid * ldb + rowid_B];
            }
          }
        }
        eleid++;
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
        // next element may cross multirows.
        T seg_rowid = binary_search_segment_number<T>(csr_row_ptr + seg_row_start, seg_row_length, eleid);
        rowid = seg_row_start + seg_rowid;
        step += csr_row_ptr[rowid + 1] - eleid;
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          sum[k] = zero;
          if (colid < N) {
            if (ORDER_ROW) {
              sum[k] += val * matB[rowid_B * ldb + colid];
            } else {
              sum[k] += val * matB[colid * ldb + rowid_B];
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
          atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
        } else {
          atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
        }
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
         int unroll_len,
         bool ORDER_ROW>
static __global__ void
csrspmm_nnz_split_aligned_kernel_v2(
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
  __shared__ U s_val[block_size / warp_size * work_size];
  __shared__ T s_col_ind[block_size / warp_size * work_size];
  __shared__ T s_row_ind[block_size / warp_size * work_size];
  __shared__ T s_seg_row_id[block_size / warp_size + 1];

  T block_segid = blockIdx.x * blockDim.y;
  T segid = block_segid + threadIdx.y;
  T element_per_seg = work_size;
  T eleid = segid * element_per_seg;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T local_segid = local_warpid;
  T block_eleid = block_segid * element_per_seg;
  T warp_laneid = local_tid & (warp_size - 1);
  U matB_buffer[unroll_len][factor] = {};
  T rowid_buffer[unroll_len] = {};
  W sum[factor] = {};
  W zero = {};


  if ((local_tid < block_size / warp_size + 1) && (block_eleid + local_tid * work_size < nnz + work_size)) {
    s_seg_row_id[local_tid] = seg_row_id[block_segid + local_tid];
  }

  __syncthreads();
   
  if (eleid >= nnz) return;

  if (warp_laneid < work_size && eleid + warp_laneid < nnz) {
    s_val[local_warpid * work_size + warp_laneid] = csr_val[eleid + warp_laneid];
    s_col_ind[local_warpid * work_size + warp_laneid] = csr_col_ind[eleid + warp_laneid];
  }
  else {
    U zero = {};
    s_val[local_warpid * work_size + warp_laneid] = zero;
    s_col_ind[local_warpid * work_size + warp_laneid] = 0;
  }

  eleid = min(eleid + warp_laneid, nnz - 1);

  T seg_row_start = s_seg_row_id[local_segid];
  T seg_row_length = s_seg_row_id[local_segid + 1] - seg_row_start;

  T seg_rowid = binary_search_segment_number<T>(csr_row_ptr + seg_row_start, seg_row_length, eleid);
  s_row_ind[local_warpid * work_size + warp_laneid] = seg_row_start + seg_rowid;


  T prev = seg_row_start;
  for (T i = 0; i < element_per_seg; i += unroll_len) {

    #pragma unroll
    for (int j = 0; j < unroll_len; j++) {
      T rowid = s_row_ind[local_warpid * work_size + i + j];
      T rowid_B = s_col_ind[local_warpid * work_size + i + j];
      U val = s_val[local_warpid * work_size + i + j];
      rowid_buffer[j] = rowid;
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        if (ORDER_ROW) {
          matB_buffer[j][k] = val * matB[rowid_B * ldb + colid];
        } else {
          matB_buffer[j][k] = val * matB[colid * ldb + rowid_B];
        }
      }
    }
    for (int j = 0; j < unroll_len; ++j) {
      T rowid = rowid_buffer[j];
      if (rowid == prev) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] += matB_buffer[j][k];
        }
      }
      else {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (ORDER_ROW) {
            atomicAdd(&matC[prev * ldc + colid], alpha * sum[k]);
          } else {
            atomicAdd(&matC[colid * ldc + prev], alpha * sum[k]);
          }
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = zero;
          sum[k] += matB_buffer[j][k];
        }
        prev = rowid;
      }
    }
  }
  #pragma unroll
  for (int k = 0; k < factor; k++) {
    T colid = block_colid + k * warp_size + warp_laneid; 
    if (ORDER_ROW) {
      atomicAdd(&matC[prev * ldc + colid], alpha * sum[k]);
    } else {
      atomicAdd(&matC[colid * ldc + prev], alpha * sum[k]);
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
         int unroll_len,
         bool ORDER_ROW>
static __global__ void
csrspmm_nnz_split_nonaligned_kernel_v2(
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
  __shared__ U s_val[block_size / warp_size * work_size];
  __shared__ T s_col_ind[block_size / warp_size * work_size];
  __shared__ T s_row_ind[block_size / warp_size * work_size];
  __shared__ T s_seg_row_id[block_size / warp_size + 1];

  T block_segid = blockIdx.x * blockDim.y;
  T segid = block_segid + threadIdx.y;
  T element_per_seg = work_size;
  T eleid = segid * element_per_seg;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T local_segid = local_warpid;
  T block_eleid = block_segid * element_per_seg;
  T warp_laneid = local_tid & (warp_size - 1);
  U matB_buffer[unroll_len][factor] = {};
  T rowid_buffer[unroll_len] = {};
  W sum[factor] = {};
  W zero = {};


  if ((local_tid < block_size / warp_size + 1) && (block_eleid + local_tid * work_size < nnz + work_size)) {
    s_seg_row_id[local_tid] = seg_row_id[block_segid + local_tid];
  }

  __syncthreads();
   
  if (eleid >= nnz) return;

  if (warp_laneid < work_size && eleid + warp_laneid < nnz) {
    s_val[local_warpid * work_size + warp_laneid] = csr_val[eleid + warp_laneid];
    s_col_ind[local_warpid * work_size + warp_laneid] = csr_col_ind[eleid + warp_laneid];
  }
  else {
    U zero = {};
    s_val[local_warpid * work_size + warp_laneid] = zero;
    s_col_ind[local_warpid * work_size + warp_laneid] = 0;
  }

  eleid = min(eleid + warp_laneid, nnz - 1);

  T seg_row_start = s_seg_row_id[local_segid];
  T seg_row_length = s_seg_row_id[local_segid + 1] - seg_row_start;

  T seg_rowid = binary_search_segment_number<T>(csr_row_ptr + seg_row_start, seg_row_length, eleid);
  s_row_ind[local_warpid * work_size + warp_laneid] = seg_row_start + seg_rowid;


  T prev = seg_row_start;
  for (T i = 0; i < element_per_seg; i += unroll_len) {

    #pragma unroll
    for (int j = 0; j < unroll_len; j++) {
      T rowid = s_row_ind[local_warpid * work_size + i + j];
      T rowid_B = s_col_ind[local_warpid * work_size + i + j];
      U val = s_val[local_warpid * work_size + i + j];
      rowid_buffer[j] = rowid;
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        if (colid < N) {
          if (ORDER_ROW) {
            matB_buffer[j][k] = val * matB[rowid_B * ldb + colid];
          } else {
            matB_buffer[j][k] = val * matB[colid * ldb + rowid_B];
          }
        }
      }
    }
    for (int j = 0; j < unroll_len; ++j) {
      T rowid = rowid_buffer[j];
      if (rowid == prev) {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] += matB_buffer[j][k];
        }
      }
      else {
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              atomicAdd(&matC[prev * ldc + colid], alpha * sum[k]);
            } else {
              atomicAdd(&matC[colid * ldc + prev], alpha * sum[k]);
            }
          }
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = zero;
          sum[k] += matB_buffer[j][k];
        }
        prev = rowid;
      }
    }
  }
  #pragma unroll
  for (int k = 0; k < factor; k++) {
    T colid = block_colid + k * warp_size + warp_laneid; 
    if (colid < N) {
      if (ORDER_ROW) {
        atomicAdd(&matC[prev * ldc + colid], alpha * sum[k]);
      } else {
        atomicAdd(&matC[colid * ldc + prev], alpha * sum[k]);
      }
    }
  }
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_eb_sr(alphasparseHandle_t handle,
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
  int warp_size, factor, work_size;
  const int block_size = RefThreadPerBlock;
  if (ORDER_ROW) {
    if (N > 128) {
      warp_size = 64;
      factor = 4;
    }
    else if (N > 64) {
      warp_size = 64;
      factor = 2;
    }
    else if (N > 32) {
      warp_size = 32;
      factor = 2;
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

  work_size = warp_size;

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T Nnzdim_worker = CEIL(nnz, (T)work_size);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  blocksPerGrid = CEIL(Nnzdim_worker, 2 * RefThreadPerBlock);
  elebalance_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    M, nnz, work_size, csr_row_ptr, seg_row_id);

  bool align = !(N % (warp_size * factor));

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);
  if (align) {
    if (warp_size == 64 && factor == 4) {
      csrspmm_nnz_split_aligned_kernel<block_size, 64, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_nnz_split_aligned_kernel<block_size, 64, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_nnz_split_aligned_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_nnz_split_aligned_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_nnz_split_aligned_kernel<block_size, 16, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
  }
  else {
    if (warp_size == 64 && factor == 4) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 64, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 64, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 16, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }

  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_eb_sr_v2(alphasparseHandle_t handle,
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
  int warp_size, factor, work_size;
  const int block_size = RefThreadPerBlock;
  if (ORDER_ROW) {
    if (N > 128) {
      warp_size = 64;
      factor = 4;
    } else if (N > 64) {
      warp_size = 64;
      factor = 2;
    } else if (N > 32) {
      warp_size = 64;
      factor = 1;
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

  work_size = warp_size;

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T Nnzdim_worker = CEIL(nnz, (T)work_size);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  blocksPerGrid = CEIL(Nnzdim_worker, 2 * RefThreadPerBlock);
  elebalance_partition_kernel<2 * RefThreadPerBlock, T>
    <<<dim3(blocksPerGrid), dim3(2 * threadsPerBlock), 0, handle->stream>>>(
    M, nnz, work_size, csr_row_ptr, seg_row_id);

  bool align = !(N % (warp_size * factor));

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);
  if (align) {
    if (warp_size == 64 && factor == 4) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 64, 64, T, U, V, W, 4, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 64, 64, T, U, V, W, 2, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 64, 64, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 32, 32, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 16, 16, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 8, 8, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_nnz_split_aligned_kernel_v2<block_size, 4, 4, T, U, V, W, 1, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
  }
  else {
    if (warp_size == 64 && factor == 4) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 64, 64, T, U, V, W, 4, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 64, 64, T, U, V, W, 2, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 64 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 64, 64, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 32, 32, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 16, 16, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 8, 8, T, U, V, W, 1, 8, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_nnz_split_nonaligned_kernel_v2<block_size, 4, 4, T, U, V, W, 1, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          seg_row_id);
    }
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
