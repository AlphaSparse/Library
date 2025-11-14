#include "alphasparse.h"

template <typename T, typename U>
__global__ static void
rot_device(
        T nnz,
        U *x_val,
        const T *x_ind,
        U *y,
        U c,
        U s
          )
{
    int tid    = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < nnz; i += stride) {
        U x_tmp = x_val[i];
        U y_tmp = y[x_ind[i]];

        x_val[i] = c * x_tmp + s * y_tmp;
        y[x_ind[i]] = c * y_tmp - s * x_tmp;
    }
}

template <typename T, typename U>
alphasparseStatus_t rot_template(alphasparseHandle_t handle,
                const void *c_coeff,
                const void *s_coeff,
                alphasparseSpVecDescr_t x,
                alphasparseDnVecDescr_t y
            )
{
const int threadPerBlock = 256;

    const T nnz          = x->nnz;
    const T blockPerGrid = min((T)2, ((T)threadPerBlock + nnz - (T)1) / threadPerBlock);
    const T *x_ind = (T *)x->idx_data;
    U *x_val = (U *)x->val_data;
    U *y_val = (U *)y->values;
    rot_device<<<dim3(blockPerGrid), dim3(threadPerBlock), 0, handle->stream>>>(nnz, x_val, x_ind, y_val, *(U*)c_coeff, *(U*)s_coeff);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseRot(alphasparseHandle_t handle,
                       const void *c_coeff,
                        const void *s_coeff,
                        alphasparseSpVecDescr_t x,
                        alphasparseDnVecDescr_t y
                )
{
    // Check for valid handle and matrix descriptor
    if (handle == nullptr) {
        return ALPHA_SPARSE_STATUS_INVALID_HANDLE;
    }

    //
    // Check the rest of pointer arguments
    //
    if (x == nullptr || y == nullptr) {
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
        return rot_template<int32_t, float>(handle, c_coeff, s_coeff, x, y);
    }
    // double real ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_R_64F) {
        return rot_template<int32_t, double>(handle, c_coeff, s_coeff, x, y);
    }
    // // single complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_32F) {
        return rot_template<int32_t, cuFloatComplex>(handle, c_coeff, s_coeff, x, y);
    }
    // double complex ; i32
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I32 && x->data_type == ALPHA_C_64F) {
        return rot_template<int32_t, cuDoubleComplex>(handle, c_coeff, s_coeff, x, y);
    }
    // single real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_32F) {
        return rot_template<int64_t, float>(handle, c_coeff, s_coeff, x, y);
    }
    // double real ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_R_64F) {
        return rot_template<int64_t, double>(handle, c_coeff, s_coeff, x, y);
    }
    // // single complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_32F) {
        return rot_template<int64_t, cuFloatComplex>(handle, c_coeff, s_coeff, x, y);
    }
    // double complex ; i64
    if (x->idx_type == ALPHA_SPARSE_INDEXTYPE_I64 && x->data_type == ALPHA_C_64F) {
        return rot_template<int64_t, cuDoubleComplex>(handle, c_coeff, s_coeff, x, y);
    }

    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}
