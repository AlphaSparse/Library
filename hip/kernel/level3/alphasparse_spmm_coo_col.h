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
  coommnn_scale_device(T M,
                      T N,
                      T K,
                      T nnz,
                      W alpha,
                      const T* __restrict__ coo_row_ptr,
                      const T* __restrict__ coo_col_ind,
                      const U* __restrict__ coo_val,
                      const U* __restrict__ matB,
                      T ldb,
                      W beta,
                      V* __restrict__ matC,
                      T ldc)
{
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < M; i += stride)
        for(int j = 0; j < N; j++)
        {
           matC[i + j * ldc] *= beta;
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
  coommnn_plain_device(T M,
                      T N,
                      T K,
                      T nnz,
                      W alpha,
                      const T* __restrict__ coo_row_ptr,
                      const T* __restrict__ coo_col_ind,
                      const U* __restrict__ coo_val,
                      const U* __restrict__ matB,
                      T ldb,
                      W beta,
                      V* __restrict__ matC,
                      T ldc)
{
  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < N; i += stride)
  {
      for (int nn = 0; nn < nnz; ++nn)
      {        
        matC[index2(i, coo_row_ptr[nn], ldc)] += alpha * coo_val[nn] * matB[index2(i, coo_col_ind[nn], ldb)];
        // printf("Caf %d : %f A %d,%d : %f B %d : %f ; %d %f\n",index2(coo_row_ptr[nn], cc, ldc), matC[index2(coo_row_ptr[nn], cc, ldc)], coo_row_ptr[nn], coo_col_ind[nn],  coo_val[nn] , index2(coo_col_ind[nn], cc, ldb), matB[index2(coo_col_ind[nn], cc, ldb)], index2( cc, coo_col_ind[nn],ldb), matB[index2(cc, coo_col_ind[nn], ldb)]);
      }
  }
}

template<typename T, typename U, typename V, typename W>
static alphasparseStatus_t
spmm_coo_col(alphasparseHandle_t handle,
             T m,
             T n,
             T k,
             T nnz,
             W alpha,
             const U* coo_val,
             const T* coo_row_ptr,
             const T* coo_col_ind,
             const U* matB,
             T ldb,
             W beta,
             V* matC,
             T ldc)
{
  const T BLOCKSIZE = 256;
  const T WF_SIZE = 8;

  coommnn_scale_device<BLOCKSIZE, WF_SIZE, T, U, V, W>
    <<<dim3(64), dim3(BLOCKSIZE), 0, handle->stream>>>
                        (m,
                         n,
                         k,
                         nnz,
                         alpha,
                         coo_row_ptr,
                         coo_col_ind,
                         coo_val,
                         matB,
                         ldb,
                         beta,
                         matC,
                         ldc);

  coommnn_plain_device<BLOCKSIZE, WF_SIZE, T, U, V, W>
    <<<dim3(64), dim3(BLOCKSIZE), 0, handle->stream>>>
                        (m,
                         n,
                         k,
                         nnz,
                         alpha,
                         coo_row_ptr,
                         coo_col_ind,
                         coo_val,
                         matB,
                         ldb,
                         beta,
                         matC,
                         ldc);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
