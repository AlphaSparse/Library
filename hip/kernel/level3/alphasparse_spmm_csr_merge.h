#include "alphasparse_spmm_utils.h"
#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include <iostream>

template <typename T>
static __forceinline__ __device__ void merge_path_search(
    const T diagonal, const T x_len, const T y_len,
    const T *__restrict__ a, const T offset_y,
    T *__restrict__ x, T *__restrict__ y)
{
    T x_min = max(diagonal - y_len, (T)0);
    T x_max = min(diagonal, x_len);
    T pivot;
    while (x_min < x_max)
    {
        pivot = (x_max + x_min) / 2;
        if (a[pivot] < offset_y + diagonal - pivot)
        {
            x_min = pivot + 1;
        }
        else
        {
            x_max = pivot;
        }
    }

    *x = min(x_min, x_len);
    *y = diagonal - *x;
}

template<int block_size,
         typename T>
__global__ void 
__launch_bounds__(512)
merge_path_search_kernel(
  T *start_xs,
  T *start_ys,
  T num_rows,
  T nnz,
  const T num_merge_items,
  const T items_per_thread,
  const T thread_num,
  const T *csr_row_ptr)
{
  __shared__ T merge_seg_ptr[block_size + 1];
  const T local_tid = threadIdx.x;
  const T tid = blockIdx.x * blockDim.x + threadIdx.x;

  const T stride = CEIL(num_rows, block_size);
  const T seg_row_idx = MIN(local_tid * stride, num_rows); 

  merge_seg_ptr[local_tid] = __ldg(&csr_row_ptr[seg_row_idx]) + seg_row_idx;
  if (local_tid == block_size - 1) merge_seg_ptr[block_size] = nnz + num_rows;

  __syncthreads();

  if (tid <= thread_num) {
    const T diagonal = MIN(items_per_thread * tid, num_merge_items);
    const T seg_start_id = binary_search_segment_number<T>(merge_seg_ptr, block_size, diagonal);

    const T seg_start_x = MIN(seg_start_id * stride, num_rows);
    const T seg_start_y = merge_seg_ptr[seg_start_id] - seg_start_x;
    const T seg_end_id = MIN(seg_start_id + 1, block_size);
    const T seg_end_x = MIN(seg_start_x + stride, num_rows);
    const T seg_end_y = merge_seg_ptr[seg_end_id] - seg_end_x;
    const T seg_diagonal = seg_start_x + seg_start_y;
    T local_start_x, local_start_y;
    const T local_diagonal = diagonal - seg_diagonal;
    merge_path_search(local_diagonal, seg_end_x - seg_start_x, seg_end_y - seg_start_y, csr_row_ptr + seg_start_x + 1,
                      seg_start_y, &local_start_x, &local_start_y);

    start_xs[tid] = seg_start_x + local_start_x;
    start_ys[tid] = seg_start_y + local_start_y;
  }
}

template<int block_size,
         int warp_size,
         int items_per_seg,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_merge_aligned_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ start_xs,
  const T* __restrict__ start_ys)
{
  const T items_per_block = block_size / warp_size * items_per_seg;
  __shared__ U s_val[items_per_block];
  __shared__ T s_col_ind[items_per_block];
  __shared__ T s_row_ptr[items_per_block];

  T segid = blockIdx.x * blockDim.y + threadIdx.y;
  T local_segid = threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  U *local_s_val = s_val + local_warpid * items_per_seg;
  T *local_s_col_ind = s_col_ind + local_warpid * items_per_seg;
  T *local_s_row_ptr = s_row_ptr + local_warpid * items_per_seg;

  if (segid * items_per_seg < M + nnz) {
    const T start_x = start_xs[segid];
    const T start_y = start_ys[segid];
    const T num_rows = start_xs[segid + 1] - start_x;
    const T num_nnz = start_ys[segid + 1] - start_y;

    if (warp_laneid < num_rows) {
      local_s_row_ptr[warp_laneid] = csr_row_ptr[start_x + warp_laneid + 1];
    }
    if (warp_laneid < num_nnz) {
      local_s_val[warp_laneid] = csr_val[start_y + warp_laneid];
      local_s_col_ind[warp_laneid] = csr_col_ind[start_y + warp_laneid];
    }

    T nnzid = start_y;
    T rowid = start_x;
    T local_x = 0;
    T local_y = 0;

    W sum[factor] = {};
    int complete_current_row = 0;
    int empty_row = 1;

    for (T i = 0; i < items_per_seg && rowid < M; i++) {
      if (local_x == num_rows || nnzid < local_s_row_ptr[local_x]) {
        T rowid_B = local_s_col_ind[local_y];

        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (ORDER_ROW) {
            sum[k] += local_s_val[local_y] * matB[rowid_B * ldb + colid];
          } else {
            sum[k] += local_s_val[local_y] * matB[colid * ldb + rowid_B];
          }
        }
        empty_row = 0;
        local_y++;
        nnzid++;
        complete_current_row = 0;
      } else {
        if (!empty_row) {
          #pragma unroll
          for (int k = 0; k < factor; k++) {
            T colid = block_colid + k * warp_size + warp_laneid; 
            if (ORDER_ROW) {
              atomicAdd(&matC[rowid * ldc + colid], alpha * sum[k]);
            } else {
              atomicAdd(&matC[colid * ldc + rowid], alpha * sum[k]);
            }
          }
          empty_row = 1;
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = {};
        }
        local_x++;
        rowid++;
        complete_current_row = 1;
      }
    }
    if (!complete_current_row) {
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
  }
}

template<int block_size,
         int warp_size,
         int items_per_seg,
         typename T,
         typename U,
         typename V,
         typename W,
         int factor,
         bool ORDER_ROW>
static __global__ void
csrspmm_merge_nonaligned_kernel(
  T M, T N, T K, T nnz, W alpha,
  const T* __restrict__ csr_row_ptr,
  const T* __restrict__ csr_col_ind,
  const U* __restrict__ csr_val,
  const U* __restrict__ matB,
  T ldb,  W beta,
  V* __restrict__ matC,
  T ldc,
  const T* __restrict__ start_xs,
  const T* __restrict__ start_ys)
{
  const T items_per_block = block_size / warp_size * items_per_seg;
  __shared__ U s_val[items_per_block];
  __shared__ T s_col_ind[items_per_block];
  __shared__ T s_row_ptr[items_per_block];

  T segid = blockIdx.x * blockDim.y + threadIdx.y;
  T local_segid = threadIdx.y;
  T block_colid = blockIdx.y * blockDim.x * factor;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T warp_laneid = local_tid & (warp_size - 1);

  U *local_s_val = s_val + local_warpid * items_per_seg;
  T *local_s_col_ind = s_col_ind + local_warpid * items_per_seg;
  T *local_s_row_ptr = s_row_ptr + local_warpid * items_per_seg;

  if (segid * items_per_seg < M + nnz) {
    const T start_x = start_xs[segid];
    const T start_y = start_ys[segid];
    const T num_rows = start_xs[segid + 1] - start_x;
    const T num_nnz = start_ys[segid + 1] - start_y;

    if (warp_laneid < num_rows) {
      local_s_row_ptr[warp_laneid] = csr_row_ptr[start_x + warp_laneid + 1];
    }
    if (warp_laneid < num_nnz) {
      local_s_val[warp_laneid] = csr_val[start_y + warp_laneid];
      local_s_col_ind[warp_laneid] = csr_col_ind[start_y + warp_laneid];
    }

    T nnzid = start_y;
    T rowid = start_x;
    T local_x = 0;
    T local_y = 0;

    W sum[factor] = {};
    int complete_current_row = 0;
    int empty_row = 1;

    for (T i = 0; i < items_per_seg && rowid < M; i++) {
      if (local_x == num_rows || nnzid < local_s_row_ptr[local_x]) {
        T rowid_B = local_s_col_ind[local_y];
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          T colid = block_colid + k * warp_size + warp_laneid; 
          if (colid < N) {
            if (ORDER_ROW) {
              sum[k] += local_s_val[local_y] * matB[rowid_B * ldb + colid];
            } else {
              sum[k] += local_s_val[local_y] * matB[colid * ldb + rowid_B];
            }
          }
        }
        empty_row = 0;
        local_y++;
        nnzid++;
        complete_current_row = 0;
      } else {
        if (!empty_row) {
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
          empty_row = 1;
        }
        #pragma unroll
        for (int k = 0; k < factor; k++) {
          sum[k] = {};
        }
        local_x++;
        rowid++;
        complete_current_row = 1;
      }
    }
    if (!complete_current_row) {
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
}
template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_merge(alphasparseHandle_t handle,
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
      warp_size = 64;
      factor = 1;
    }
    else if (N > 16) {
      warp_size = 32;
      factor = 1;
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

  bool align = !(N % (warp_size * factor));

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T merge_length = nnz + M;
  T Pathdim_worker = CEIL(merge_length, work_size);
  T *start_xs, *start_ys;
  start_xs = reinterpret_cast<T *>(externalBuffer);
  start_ys = reinterpret_cast<T *>(start_xs + Pathdim_worker + 1);

  merge_path_search_kernel<2 * RefThreadPerBlock, T><<<CEIL((Pathdim_worker + 1), 2 * RefThreadPerBlock), 2 * RefThreadPerBlock, 0, handle->stream>>>(
    start_xs,
    start_ys,
    M,
    nnz,
    merge_length,
    work_size,
    Pathdim_worker,
    csr_row_ptr);

  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size * factor);
  T Ndim_thread_per_tb = warp_size;
  T Pathdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Pathdim_threadblock = CEIL(Pathdim_worker, Pathdim_thread_per_tb);


  dim3 gridDim(Pathdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Pathdim_thread_per_tb, 1);
  if (align) {
    if (warp_size == 64 && factor == 4) {
      csrspmm_merge_aligned_kernel<block_size, 64, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_merge_aligned_kernel<block_size, 64, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 64 && factor == 1) {
      csrspmm_merge_aligned_kernel<block_size, 64, 64, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_merge_aligned_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_merge_aligned_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_merge_aligned_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_merge_aligned_kernel<block_size, 16, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_merge_aligned_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_merge_aligned_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_merge_aligned_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
  }
  else {
    if (warp_size == 64 && factor == 4) {
      csrspmm_merge_nonaligned_kernel<block_size, 64, 64, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 64 && factor == 2) {
      csrspmm_merge_nonaligned_kernel<block_size, 64, 64, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 64 && factor == 1) {
      csrspmm_merge_nonaligned_kernel<block_size, 64, 64, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 32 && factor == 4) {
      csrspmm_merge_nonaligned_kernel<block_size, 32, 32, T, U, V, W, 4, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 32 && factor == 2) {
      csrspmm_merge_nonaligned_kernel<block_size, 32, 32, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 32 && factor == 1) {
      csrspmm_merge_nonaligned_kernel<block_size, 32, 32, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 16 && factor == 2) {
      csrspmm_merge_nonaligned_kernel<block_size, 16, 16, T, U, V, W, 2, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 16 && factor == 1) {
      csrspmm_merge_nonaligned_kernel<block_size, 16, 16, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 8 && factor == 1) {
      csrspmm_merge_nonaligned_kernel<block_size, 8, 8, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }
    if (warp_size == 4 && factor == 1) {
      csrspmm_merge_nonaligned_kernel<block_size, 4, 4, T, U, V, W, 1, ORDER_ROW>
        <<<gridDim, blockDim, 0, handle->stream>>>(
          M, N, K, nnz, alpha,
          csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
          beta, matC, ldc,
          start_xs, start_ys);
    }

  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
