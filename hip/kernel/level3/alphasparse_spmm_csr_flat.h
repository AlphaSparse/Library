#include "alphasparse_spmm_utils.h"
#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include <iostream>

template<typename T>
static __global__ void
flat_partition_kernel(
  T M, T nnz, T items_per_seg,
  const T* __restrict__ csr_row_ptr,
  T* seg_row_id)
{
  T tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < M) {
    T cur_row_seg_id = CEIL(csr_row_ptr[tid] , items_per_seg);
    T next_row_seg_id = CEIL(csr_row_ptr[tid + 1] , items_per_seg);
    if (cur_row_seg_id != next_row_seg_id) {
      for (T i = cur_row_seg_id; i < next_row_seg_id; i++) {
        seg_row_id[i] = tid;
      }
    }
  } else if (tid == M) {
    seg_row_id[CEIL(nnz, items_per_seg)] = M;
  }
}

template<int block_size,
         int warp_size,
         int work_size,
         typename T,
         typename U,
         typename V,
         typename W,
         bool ORDER_ROW>
static __global__ void
csrspmm_flat_compute_kernel(
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
  constexpr int block_work_size = block_size / warp_size * work_size;
  __shared__ U s_val[block_work_size];
  __shared__ T s_col_ind[block_work_size];
  __shared__ V s_sum[block_size * work_size];
  __shared__ T s_block_row_id[2];

  T blockid = blockIdx.x;

  T local_tid = threadIdx.y * blockDim.x + threadIdx.x;
  T local_warpid = local_tid / warp_size;
  T block_eleid = blockIdx.x * blockDim.y * work_size;
  T warp_laneid = local_tid & (warp_size - 1);
  T colid = blockIdx.y * blockDim.x + warp_laneid;

  // load nnzs to LDS.
  for (T i = local_tid; i < block_work_size; i += block_size) {
    if (block_eleid + i < nnz) {
      s_val[i] = csr_val[block_eleid + i];
      s_col_ind[i] = csr_col_ind[block_eleid + i];
    }
  }

  //load seg_row_id to LDS.
  // if (local_tid < 2) {
  //   s_block_row_id[local_tid] = seg_row_id[blockid + local_tid];
  // }

  __syncthreads();

  // compute partial sum and store to LDS.
  for (T i = local_warpid; i < block_work_size; i += block_size / warp_size) {
    if (block_eleid + i < nnz && colid < N) {
      T matB_rowid = s_col_ind[i];
      if (ORDER_ROW) {
        s_sum[i * warp_size + warp_laneid] = s_val[i] * matB[matB_rowid * ldb + colid];
      } else {
        s_sum[i * warp_size + warp_laneid] = s_val[i] * matB[colid * ldb + matB_rowid];
      }
    }
  }

  __syncthreads();

  //every warp reduces partial sums from a same row.
  T block_start_row_id = seg_row_id[blockid];
  T block_end_row_id = seg_row_id[blockid + 1];
  if (block_start_row_id == block_end_row_id || csr_row_ptr[block_end_row_id] % block_work_size != 0) {
    block_end_row_id = MIN(block_end_row_id + 1, M);
  }

  for (T rowid = block_start_row_id + local_warpid; rowid < block_end_row_id; rowid += block_size / warp_size) {
    T row_start_eleid = csr_row_ptr[rowid];
    T row_end_eleid = csr_row_ptr[rowid + 1];
    T local_reduce_start_eleid = MAX(row_start_eleid, block_eleid) - block_eleid;
    T local_reduce_end_eleid = MIN(row_end_eleid, block_eleid + block_work_size) - block_eleid;
    W sum = {};
    for (T i = local_reduce_start_eleid; i< local_reduce_end_eleid; i++) {
      sum += s_sum[i * warp_size + warp_laneid];
    }
    if (colid < N) {
      if (ORDER_ROW) {
        atomicAdd(&matC[rowid * ldc + colid], alpha * sum);
      } else {
        atomicAdd(&matC[colid * ldc + rowid], alpha * sum);
      }
    }
  }
}

template<typename T, typename U, typename V, typename W, bool ORDER_ROW>
alphasparseStatus_t
csrspmm_flat(alphasparseHandle_t handle,
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
  int warp_size, work_size;
  const int block_size = RefThreadPerBlock;
  if (ORDER_ROW) {
    if (N > 8) {
      warp_size = 16;
      work_size = 16;
    } else if (N > 4) {
      warp_size = 8;
      work_size = 8;
    } else {
      warp_size = 4;
      work_size = 4;
    }

  } else {
    if (N > 4) {
      warp_size = 8;
      work_size = 8;
    } else {
      warp_size = 4;
      work_size = 4;
    }
  }

  T threadsPerBlock = RefThreadPerBlock;
  T blocksPerGrid = CEIL(M * N, threadsPerBlock);
  matrix_scalar_multiply_kernel<T, V, W>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, N, beta, matC);

  T *seg_row_id;
  seg_row_id = reinterpret_cast<T *>(externalBuffer);
  blocksPerGrid = CEIL(M + 1, RefThreadPerBlock);
  flat_partition_kernel<T>
    <<<dim3(blocksPerGrid), dim3(threadsPerBlock), 0, handle->stream>>>(
    M, nnz, block_size / warp_size * work_size, csr_row_ptr, seg_row_id);

  T Nnzdim_worker = CEIL(nnz, (T)work_size);
  T Ndim_worker = N;
  T Ndim_threadblock = CEIL(Ndim_worker, warp_size);
  T Ndim_thread_per_tb = warp_size;
  T Nnzdim_thread_per_tb = CEIL(RefThreadPerBlock, Ndim_thread_per_tb);
  T Nnzdim_threadblock = CEIL(Nnzdim_worker, Nnzdim_thread_per_tb);

  dim3 gridDim(Nnzdim_threadblock, Ndim_threadblock, 1);
  dim3 blockDim(Ndim_thread_per_tb, Nnzdim_thread_per_tb, 1);
  if (warp_size == 16) {
    csrspmm_flat_compute_kernel<block_size, 16, 16, T, U, V, W, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 8) {
    csrspmm_flat_compute_kernel<block_size, 8, 8, T, U, V, W, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  if (warp_size == 4) {
    csrspmm_flat_compute_kernel<block_size, 4, 4, T, U, V, W, ORDER_ROW>
      <<<gridDim, blockDim, 0, handle->stream>>>(
        M, N, K, nnz, alpha,
        csr_row_ptr, csr_col_ind, csr_val, matB, ldb,
        beta, matC, ldc,
        seg_row_id);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
