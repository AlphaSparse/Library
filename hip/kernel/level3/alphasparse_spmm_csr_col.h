#include "hip/hip_runtime.h"
#include "alphasparse.h"

template<int BLOCKSIZE,
         int WF_SIZE,
         typename T,
         typename U,
         typename V,
         typename W>
static __global__ void
__launch_bounds__(BLOCKSIZE)
  csrmmnn_general_device(T M,
                         T N,
                         T K,
                         T nnz,
                         W alpha,
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

  T colB = col * ldb;
  T colC = col * ldc;

  V zero = {};

  __shared__ T shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
  __shared__ V shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

  // one wavefront process one row
  for (T row = gid / WF_SIZE; row < M; row += nwf) {
    const T row_start = csr_row_ptr[row];
    const T row_end = csr_row_ptr[row + 1];

    V sum = zero;

    for (T j = row_start; j < row_end; j += WF_SIZE) {
      T k = j + lid;

      // __syncthreads();
      V csr_val_k = {};
      csr_val_k += csr_val[k];
      shared_col[wid][lid] = (k < row_end) ? csr_col_ind[k] : 0;
      shared_val[wid][lid] = (k < row_end) ? csr_val_k : zero;

      // __syncthreads();

      if (col >= N)
        continue;

      for (T i = 0; i < WF_SIZE; ++i) {
        sum += shared_val[wid][i] * matB[shared_col[wid][i] + colB];
      }
    }

    if (col < N) {
      matC[row + colC] = alpha * sum + beta * matC[row + colC];
    }
  }
}

template<int BLOCKSIZE,
         int WF_SIZE,
         typename T,
         typename U,
         typename V,
         typename W>
static __global__ void
__launch_bounds__(BLOCKSIZE)
  csrmmnn_general_unroll_stride4_device(T M,
                                        T N,
                                        T K,
                                        T nnz,
                                        W alpha,
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
  T col = lid + blockIdx.y * WF_SIZE * 4;

  T colB = col * ldb;
  T colC = col * ldc;

  V zero = {};

  __shared__ T shared_col[BLOCKSIZE / WF_SIZE][WF_SIZE];
  __shared__ V shared_val[BLOCKSIZE / WF_SIZE][WF_SIZE];

  // one wavefront process one row
  for (T row = gid / WF_SIZE; row < M; row += nwf) {
    const T row_start = csr_row_ptr[row];
    const T row_end = csr_row_ptr[row + 1];

    V sum[4] = { zero, zero, zero, zero };

    T j = row_start;
    for (; j < row_end; j += WF_SIZE) {
      T k0 = j + lid;

      // __syncthreads();

      shared_col[wid][lid] = (k0 < row_end) ? csr_col_ind[k0] : 0;
      shared_val[wid][lid] = (k0 < row_end) ? csr_val[k0] : zero;

      // __syncthreads();

      if (col >= N)
        continue;

      if (blockIdx.y == gridDim.x - 1) {
        for (T i = 0; i < WF_SIZE; ++i) {
          sum[0] += shared_val[wid][i] * matB[shared_col[wid][i] + colB];
          if (col + WF_SIZE < N)
            sum[1] += shared_val[wid][i] *
                      matB[shared_col[wid][i] + colB + WF_SIZE * ldb];
          if (col + WF_SIZE * 2 < N)
            sum[2] += shared_val[wid][i] *
                      matB[shared_col[wid][i] + colB + WF_SIZE * 2 * ldb];
          if (col + WF_SIZE * 3 < N)
            sum[3] += shared_val[wid][i] *
                      matB[shared_col[wid][i] + colB + WF_SIZE * 3 * ldb];
        }
      } else {
        for (T i = 0; i < WF_SIZE; ++i) {
          sum[0] += shared_val[wid][i] * matB[shared_col[wid][i] + colB];
          sum[1] += shared_val[wid][i] *
                    matB[shared_col[wid][i] + colB + WF_SIZE * ldb];
          sum[2] += shared_val[wid][i] *
                    matB[shared_col[wid][i] + colB + WF_SIZE * 2 * ldb];
          sum[3] += shared_val[wid][i] *
                    matB[shared_col[wid][i] + colB + WF_SIZE * 3 * ldb];
        }
      }
    }

    if (col < N) {
      matC[row + colC] = alpha * sum[0] + beta * matC[row + colC];
    }
    if (col + WF_SIZE < N) {
      matC[row + colC + WF_SIZE * ldc] =
        alpha * sum[1] + beta * matC[row + colC + WF_SIZE * ldc];
    }
    if (col + 2 * WF_SIZE < N) {
      matC[row + colC + WF_SIZE * 2 * ldc] =
        alpha * sum[2] + beta * matC[row + colC + WF_SIZE * 2 * ldc];
    }
    if (col + 3 * WF_SIZE < N) {
      matC[row + colC + WF_SIZE * 3 * ldc] =
        alpha * sum[3] + beta * matC[row + colC + WF_SIZE * 3 * ldc];
    }
  }
}

template<typename T, typename U, typename V, typename W>
static alphasparseStatus_t
spmm_csr_col(alphasparseHandle_t handle,
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
  const T WF_SIZE = 8;
  csrmmnn_general_device<BLOCKSIZE, WF_SIZE, T, U, V, W>
    <<<dim3((WF_SIZE * m - 1) / BLOCKSIZE + 1, (n - 1) / WF_SIZE + 1),
       dim3(BLOCKSIZE),
       0,
       handle->stream>>>(m,
                         n,
                         k,
                         nnz,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         matB,
                         ldb,
                         beta,
                         matC,
                         ldc);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
