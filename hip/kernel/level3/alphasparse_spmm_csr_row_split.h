#include "alphasparse_spmm_utils.h"
#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include <iostream>

template<int block_size,
         int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_row_split_aligned_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];

  T rowid = blockIdx.x * blockDim.y + threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;
  T local_offset = threadIdx.y * blockDim.x;
  T local_tid = local_offset + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  W sum[factor] = {};
  if (rowid >= M) return;
  T rowstart = csr_row_ptr[rowid];
  T rowend = csr_row_ptr[rowid + 1];
  T eleid = rowstart;
  for (T i = rowstart; i < rowend; i += warp_size) {
    if (eleid + warp_laneid < rowend) {
      s_val[local_tid] = csr_val[eleid + warp_laneid];
      s_col_ind[local_tid] = csr_col_ind[eleid + warp_laneid];
    }
    else {
      U zero = {};
      s_val[local_tid] = zero;
      s_col_ind[local_tid] = 0;
    }

    for (T j = 0; j < warp_size; j++) {
      T rowid_B = s_col_ind[local_offset + j];
      U val = s_val[local_offset + j];

      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * warp_size + warp_laneid; 
        if (ORDER_ROW) {
          sum[k] += val * matB[rowid_B * ldb + colid];
        }
        else {
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
      matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
    }
    else {
      matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
    }
  }
}

template<int block_size,
         int warp_size,
         int subwarp_size, 
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_row_split_aligned_kernel_v2(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];

  T rowid = blockIdx.x * blockDim.y + threadIdx.y;
  T block_colid = blockIdx.y * subwarp_size * factor;
  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_offset = local_tid - local_tid % subwarp_size;
  T local_subwarpid = threadIdx.x / subwarp_size;
  T subwarp_laneid = local_tid & (subwarp_size - 1);

  W sum[factor] = {};
  if (rowid >= M) return;
  T rowstart = csr_row_ptr[rowid];
  T rowend = csr_row_ptr[rowid + 1];
  for (T i = rowstart + local_subwarpid * subwarp_size; i < rowend; i += warp_size) {
    if (i + subwarp_laneid < rowend) {
      s_val[local_tid] = csr_val[i + subwarp_laneid];
      s_col_ind[local_tid] = csr_col_ind[i + subwarp_laneid];
    }
    else {
      U zero = {};
      s_val[local_tid] = zero;
      s_col_ind[local_tid] = 0;
    }

    for (T j = 0; j < subwarp_size; j++) {
      T rowid_B = s_col_ind[local_offset + j];
      U val = s_val[local_offset + j];

      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * subwarp_size + subwarp_laneid; 
        if (ORDER_ROW) {
          sum[k] += val * matB[rowid_B * ldb + colid];
        }
        else {
          sum[k] += val * matB[colid * ldb + rowid_B];
        }
      }
    }
  }
  if constexpr (warp_size == subwarp_size) {
    #pragma unroll
    for (int k = 0; k < factor; k++) {
      sum[k] = warpReduce<W, warp_size, subwarp_size>(sum[k]);
    }

    if (local_subwarpid == 0) {
      #pragma unroll
      for (int k = 0; k < factor; k++) {
        T colid = block_colid + k * subwarp_size + subwarp_laneid; 
        if (ORDER_ROW) {
          matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
        }
        else {
          matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
        }
      }
    }
  }
  else {
    #pragma unroll
    for (int k = 0; k < factor; k++) {
      T colid = block_colid + k * warp_size + subwarp_laneid; 
      if (ORDER_ROW) {
        matC[rowid * ldc + colid] = alpha * sum[k] + beta * matC[rowid * ldc + colid];
      }
      else {
        matC[colid * ldc + rowid] = alpha * sum[k] + beta * matC[colid * ldc + rowid];
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
csrspmm_row_split_nonaligned_kernel(
  T M, T N, T K, T nnz, W alpha, 
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc)
{
  __shared__ U s_val[block_size];
  __shared__ T s_col_ind[block_size];

  T rowid = blockIdx.x * blockDim.y + threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;
  T local_offset = threadIdx.y * blockDim.x;
  T local_tid = local_offset + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  W sum[factor] = {};
  if (rowid >= M) return;
  T rowstart = csr_row_ptr[rowid];
  T rowend = csr_row_ptr[rowid + 1];
  T eleid = rowstart;
  for (T i = rowstart; i < rowend; i += warp_size) {
    if (eleid + warp_laneid < rowend) {
      s_val[local_tid] = csr_val[eleid + warp_laneid];
      s_col_ind[local_tid] = csr_col_ind[eleid + warp_laneid];
    }
    else {
      U zero = {};
      s_val[local_tid] = zero;
      s_col_ind[local_tid] = 0;
    }

    for (T j = 0; j < warp_size; j++) {
      T rowid_B = s_col_ind[local_offset + j];
      U val = s_val[local_offset + j];
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

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_rb_sr(alphasparseHandle_t handle,
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
  const int block_size = RefThreadPerBlock;
  T Mdim_worker = M;
  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Mdim_thread_per_tb = CEIL(block_size, Ndim_thread_per_tb);
  T Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);
  if (align) {
    if (warp_size == 64 && factor == 4) {
      csrspmm_row_split_aligned_kernel<block_size, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_row_split_aligned_kernel<block_size, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_row_split_aligned_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_row_split_aligned_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_row_split_aligned_kernel<block_size, 32, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 16 && factor == 4) {
      csrspmm_row_split_aligned_kernel<block_size, 16, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_row_split_aligned_kernel<block_size, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_row_split_aligned_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_row_split_aligned_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_row_split_aligned_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
  }
  else {
    if (warp_size == 64 && factor == 4) {
      csrspmm_row_split_nonaligned_kernel<block_size, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_row_split_nonaligned_kernel<block_size, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_row_split_nonaligned_kernel<block_size, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_row_split_nonaligned_kernel<block_size, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_row_split_nonaligned_kernel<block_size, 32, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 16 && factor == 4) {
      csrspmm_row_split_nonaligned_kernel<block_size, 16, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_row_split_nonaligned_kernel<block_size, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_row_split_nonaligned_kernel<block_size, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_row_split_nonaligned_kernel<block_size, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_row_split_nonaligned_kernel<block_size, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc);
    }
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_rb_sr_v2(alphasparseHandle_t handle,
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
  // int warp_size, factor;
  // if (ORDER_ROW) {
  //   if (N >= 128) {
  //     warp_size = 32;
  //     factor = 4;
  //   }
  //   else if (N >= 64) {
  //     warp_size = 32;
  //     factor = 2;
  //   }
  //   else if (N >= 32) {
  //     warp_size = 32;
  //     factor = 1;
  //   }
  //   else if (N >= 16) {
  //     warp_size = 16;
  //     factor = 1;
  //   }
  //   else if (N >= 8) {
  //     warp_size = 8;
  //     factor = 1;
  //   }
  //   else {
  //     warp_size = 4;
  //     factor = 1;
  //   }
  // }
  // else {
  //   if (N > 4) {
  //     warp_size = 8;
  //     factor = 1;
  //   } else {
  //     warp_size = 4;
  //     factor = 1;
  //   }
  // }
  int warp_size = 16;
  int subwarp_size = 16;
  int factor = 1;
  const int block_size = RefThreadPerBlock;
  T Mdim_worker = M;
  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, subwarp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Mdim_thread_per_tb = CEIL(block_size, Ndim_thread_per_tb);
  T Mdim_threadblock = CEIL(Mdim_worker, Mdim_thread_per_tb);

  dim3 gridDim(Mdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Mdim_thread_per_tb, 1);
  csrspmm_row_split_aligned_kernel_v2<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
    <<<gridDim, blockDim, 0, handle->stream>>>(
      M, N, K, nnz, alpha,
      csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
      beta, matC, ldc);
  // if (warp_size == 32 && factor == 4) {
  //   csrspmm_row_split_aligned_kernel_v2<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
  //     <<<gridDim, blockDim, 0, handle->stream>>>(
  //       M, N, K, nnz, alpha,
  //       csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
  //       beta, matC, ldc);
  // }
  // if (warp_size == 32 && factor == 2) {
  //   csrspmm_row_split_aligned_kernel_v2<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
  //     <<<gridDim, blockDim, 0, handle->stream>>>(
  //       M, N, K, nnz, alpha,
  //       csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
  //       beta, matC, ldc);
  // }
  // if (warp_size == 32 && factor == 1) {
  //   csrspmm_row_split_aligned_kernel_v2<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
  //     <<<gridDim, blockDim, 0, handle->stream>>>(
  //       M, N, K, nnz, alpha,
  //       csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
  //       beta, matC, ldc);
  // }
  // if (warp_size == 16 && factor == 1) {
  //   csrspmm_row_split_aligned_kernel_v2<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
  //     <<<gridDim, blockDim, 0, handle->stream>>>(
  //       M, N, K, nnz, alpha,
  //       csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
  //       beta, matC, ldc);
  // }
  // if (warp_size == 8 && factor == 1) {
  //   csrspmm_row_split_aligned_kernel_v2<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
  //     <<<gridDim, blockDim, 0, handle->stream>>>(
  //       M, N, K, nnz, alpha,
  //       csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
  //       beta, matC, ldc);
  // }
  // if (warp_size == 4 && factor == 1) {
  //   csrspmm_row_split_aligned_kernel_v2<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
  //     <<<gridDim, blockDim, 0, handle->stream>>>(
  //       M, N, K, nnz, alpha,
  //       csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
  //       beta, matC, ldc);
  // }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
