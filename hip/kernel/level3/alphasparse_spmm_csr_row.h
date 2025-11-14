#include "hip/hip_runtime.h"

#include "alphasparse.h"
#include <iostream>

// static const int BLOCKSIZE = 256;
// static const int WF_SIZE = 8;

template<int BLOCKSIZE,
         int WF_SIZE,
         typename T,
         typename U,
         typename V,
         typename W>
static __global__ void
__launch_bounds__(BLOCKSIZE)
  csrmmnn_stream_device(T block_rows,
                        T N,
                        T K,
                        T nnz,
                        W alpha,
                        const T* __restrict__ row_block,
                        const T* __restrict__ csr_row_ptr,
                        const T* __restrict__ csr_col_ind,
                        const U* __restrict__ csr_val,
                        const U* __restrict__ matB,
                        T ldb,
                        W beta,
                        V* __restrict__ matC,
                        T ldc)
{
  T tid = threadIdx.x;
  T gid = blockIdx.x * BLOCKSIZE + tid;
  T lid = gid & (WF_SIZE - 1);
  T wid = tid / WF_SIZE;
  T nwf = gridDim.x * BLOCKSIZE / WF_SIZE;
  T col = lid + blockIdx.y * WF_SIZE;

  V zero = {};

  __shared__ T shared_row[BLOCKSIZE];
  __shared__ T shared_col[BLOCKSIZE];
  __shared__ V shared_val[BLOCKSIZE];

  const T start_row_id = row_block[blockIdx.x];
  const T end_row_id = row_block[blockIdx.x + 1];
  const T block_row_start = csr_row_ptr[start_row_id];
  const T block_row_end = csr_row_ptr[end_row_id];
  const T num_rows = end_row_id - start_row_id;
  const T block_nnz = block_row_start - block_row_end;
  const T wf_per_block = BLOCKSIZE / WF_SIZE;

  T k = block_row_start + tid;

  __syncthreads();
  if (k < block_row_end){
    shared_col[tid] = csr_col_ind[k];
    shared_val[tid] = csr_val[k];
  }
  else{
    shared_col[tid] = 0;
    shared_val[tid] = zero;
  }
  __syncthreads();

  // one wavefront process one row
  for (T row = start_row_id + wid; row < end_row_id; row += wf_per_block) {
    const T row_start = csr_row_ptr[row];
    const T row_end = csr_row_ptr[row + 1];

    V sum = zero;

    // T k = block_row_start + tid;

    // __syncthreads();
    // shared_col[tid] = (k < block_row_end) ? csr_col_ind[k] : 0;
    // shared_val[tid] = (k < block_row_end) ? csr_val[k] : zero;
    // __syncthreads();

    if (col >= N)
      continue;

    for (T j = row_start; j < row_end; ++j) {
      T i = j - block_row_start;
      sum += shared_val[i] * matB[shared_col[i] * ldb + col];
    }

    // if (col < N) {
    // matC[row + colC] = alpha * sum + beta * matC[row + colC];
    matC[row * ldc + col] = alpha * sum + beta * matC[row * ldc + col];
    // }
  }
}

template<typename T>
static void
get_row_block(T block_size,
              T m,
              const T* __restrict__ csr_row_ptr,
              std::vector<T>& row_block)
{
  // TODO if  row_nnz > block_size

  T cur_size = 0;
  row_block.push_back(0);
  for (int r = 0; r < m; r++) {
    T row_nnz = csr_row_ptr[r + 1] - csr_row_ptr[r];
    if (cur_size + row_nnz <= block_size) {
      cur_size += row_nnz;
    } else {
      cur_size = row_nnz;
      row_block.push_back(r);
    }
  }
  row_block.push_back(m);
}

template<typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmm_csr_row(alphasparseHandle_t handle,
             T m,
             T n,
             T k,
             T nnz,
             W alpha,
             const U* csr_val,
             const T* csr_row_ptr,
             const T* csr_col_ind,
             const U* matB,
             T ldb,
             W beta,
             V* matC,
             T ldc)
{
  const T BLOCKSIZE = 256;
  const T WF_SIZE = 16;

  double time;
  time = get_time_us();

  std::vector<T> hptr(m + 1);
  hipMemcpyAsync(hptr.data(),
                  csr_row_ptr,
                  sizeof(T) * (m + 1),
                  hipMemcpyDeviceToHost,
                  handle->stream);
  hipStreamSynchronize(handle->stream);

  std::vector<T> row_block;
  get_row_block(BLOCKSIZE, m, hptr.data(), row_block);
  T* d_row_block;
  T row_block_size = row_block.size();
  hipMalloc(&d_row_block, sizeof(T) * row_block_size);
  hipMemcpyAsync(d_row_block,
                  row_block.data(),
                  sizeof(T) * row_block_size,
                  hipMemcpyHostToDevice,
                  handle->stream);
  hipStreamSynchronize(handle->stream);

  time = (get_time_us() - time) / (1e3);
  std::cout << "get bin time: " << time << std::endl;

  // // DEBUGGGGGG
  // T nnz_sum = 0;
  // for (int i = 0; i < row_block.size() - 1; i++) {
  //     T nnz = csr_row_ptr[row_block[i+1]] - csr_row_ptr[row_block[i]];
  //     nnz_sum += nnz;
  //     printf("%d, %d, %d\n", row_block[i], nnz, nnz_sum);
  // }
  T size = row_block_size - 1;
  // hipLaunchKernelGGL((csrmmnn_stream_device<BLOCKSIZE, WF_SIZE>),
  // dim3((WF_SIZE * size - 1) / BLOCKSIZE + 1, CEIL(n, WF_SIZE)),
  // dim3(BLOCKSIZE), 0, handle->stream, size, n, k, nnz, alpha, d_row_block,
  // csr_row_ptr, csr_col_ind, csr_val, matB, ldb, beta, matC, ldc);
  csrmmnn_stream_device<BLOCKSIZE, WF_SIZE, T, U, V, W>
    <<<dim3(size, CEIL(n, WF_SIZE)), dim3(BLOCKSIZE), 0, handle->stream>>>(
      size,
      n,
      k,
      nnz,
      alpha,
      d_row_block,
      csr_row_ptr,
      csr_col_ind,
      csr_val,
      matB,
      ldb,
      beta,
      matC,
      ldc);
  // TODO free
  return ALPHA_SPARSE_STATUS_SUCCESS;
}