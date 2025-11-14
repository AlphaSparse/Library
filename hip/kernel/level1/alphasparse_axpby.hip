
#include "alphasparse.h"

template <typename T, typename U, typename V>
__global__ static void
mul_device(const T size,
        const T nnz,
      const V alpha,
      const U *x_val,
      const V beta,
      const T *x_ind,
      U *y)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (T i = tid; i < size; i += stride) {
        y[i] = beta * y[i];
    }
}

template <typename T, typename U, typename V>
__global__ static void
axpby_device(const T size,
        const T nnz,
      const V alpha,
      const U *x_val,
      const V beta,
      const T *x_ind,
      U *y)
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;

    for (T i = tid; i < nnz; i += stride) {
        y[x_ind[i]] = alpha * x_val[i] + y[x_ind[i]];
    }
}

const int threadPerBlock = 256;

template <typename T, typename U, typename V>
alphasparseStatus_t axpby_template(alphasparseHandle_t handle,
            const void *alpha,
            const alphasparseSpVecDescr_t x,
            const void *beta,
            alphasparseDnVecDescr_t y)
{
    const T size          = x->size;
    const T nnz          = x->nnz;
    const T blockPerGrid = min((T)2, ((T)threadPerBlock + nnz - (T)1) / threadPerBlock);
    const T *x_ind = (T *)x->idx_data;
    const U *x_val = (U *)x->val_data;
    U *y_val = (U *)y->values;
    const V alpha_val = *(V*)alpha;
    const V beta_val = *(V*)beta;
    mul_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(size, nnz, alpha_val, x_val, beta_val, x_ind, y_val);
    axpby_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(size, nnz, alpha_val, x_val, beta_val, x_ind, y_val);
   
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseAxpby(alphasparseHandle_t handle,
                      const void *alpha,
                      const alphasparseSpVecDescr_t x,
                      const void *beta,
                      alphasparseDnVecDescr_t y)
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }
    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr || alpha == nullptr || beta == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_POINTER;
    }

    // Check if descriptors are initialized
    if (x->init == false || y->init == false) {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    // Check for matching types while we do not support mixed precision computation
    if (x->data_type != y->data_type) {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }

    // single real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_32F) {
        return axpby_template<int32_t, float, float>(handle, alpha, x, beta, y);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_64F) {
        return axpby_template<int32_t, double, double>(handle, alpha, x, beta, y);
    }
    // // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_32F) {
        return axpby_template<int32_t, cuFloatComplex, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_64F) {
        return axpby_template<int32_t, cuDoubleComplex, cuDoubleComplex>(handle, alpha, x, beta, y);
    }
    // single real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_32F) {
        return axpby_template<int64_t, float, float>(handle, alpha, x, beta, y);
    }
    // double real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_64F) {
        return axpby_template<int64_t, double, double>(handle, alpha, x, beta, y);
    }
    // // single complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_32F) {
        return axpby_template<int64_t, cuFloatComplex, cuFloatComplex>(handle, alpha, x, beta, y);
    }
    // double complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_64F) {
        return axpby_template<int64_t, cuDoubleComplex, cuDoubleComplex>(handle, alpha, x, beta, y);
    }

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}
