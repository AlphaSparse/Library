#include "alphasparse_spgemm_csr.h"
#include "alphasparse_spgemm_nnz_csr.h"
#include <iostream>

template<typename T, typename U>
alphasparseStatus_t
spgemm_reuse_template(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      alphasparseOperation_t opB,
                      const void* alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseSpMatDescr_t matB,
                      const void* beta,
                      alphasparseSpMatDescr_t matC,
                      alphasparseSpGEMMDescr_t spgemmDescr)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {
      if (spgemmDescr->spgemm_reuse_flag) {
        T* dCptr = NULL;
        T* dCcol = NULL;
        U* dCval = NULL;
        hipMalloc((void**)&dCptr, sizeof(T) * (matA->rows + 1));
        hipMemcpy(dCptr,
                   matC->row_data,
                   (matC->rows + 1) * sizeof(T),
                   hipMemcpyDeviceToHost);
        hipMalloc((void**)&dCcol, sizeof(T) * matC->nnz);
        hipMalloc((void**)&dCval, sizeof(U) * matC->nnz);
        bool mul = true;
        bool add = false;
        spgemm_csr<T, U>(handle,
                         matC->rows,
                         std::max(matC->cols, matA->cols),
                         *((U*)alpha),
                         (T*)matA->row_data,
                         (T*)matA->col_data,
                         (U*)matA->val_data,
                         (T*)matB->row_data,
                         (T*)matB->col_data,
                         (U*)matB->val_data,
                         *((U*)beta),
                         (T*)matC->row_data,
                         (T*)matC->col_data,
                         (U*)matC->val_data,
                         dCptr,
                         dCcol,
                         dCval,
                         matA->idx_base,
                         matB->idx_base,
                         matC->idx_base,
                         matC->idx_base,
                         mul,
                         add);

        matC->val_data = dCval;
        matC->row_data = dCptr;
        matC->col_data = dCcol;
      } else {
        T nnz_C;
        T* dCptr = NULL;
        T* dCcol = NULL;
        U* dCval = NULL;
        hipMalloc((void**)&dCptr, sizeof(T) * (matA->rows + 1));
        spgemm_nnz_csr(handle,
                       (T)matC->rows,
                       (T)matC->cols,
                       (T)matA->cols,
                       (T)matA->nnz,
                       matA->row_data,
                       matA->col_data,
                       (T)matB->nnz,
                       matB->row_data,
                       matB->col_data,
                       (T)matC->nnz,
                       matC->row_data,
                       matC->col_data,
                       dCptr,
                       &nnz_C);

        hipMalloc((void**)&dCcol, sizeof(T) * nnz_C);
        hipMalloc((void**)&dCval, sizeof(U) * nnz_C);
        bool mul = true;
        bool add = false;
        spgemm_csr<T, U>(handle,
                         matC->rows,
                         std::max(matC->cols, matA->cols),
                         *((U*)alpha),
                         (T*)matA->row_data,
                         (T*)matA->col_data,
                         (U*)matA->val_data,
                         (T*)matB->row_data,
                         (T*)matB->col_data,
                         (U*)matB->val_data,
                         *((U*)beta),
                         (T*)matC->row_data,
                         (T*)matC->col_data,
                         (U*)matC->val_data,
                         dCptr,
                         dCcol,
                         dCval,
                         matA->idx_base,
                         matB->idx_base,
                         matC->idx_base,
                         matC->idx_base,
                         mul,
                         add);

        matC->val_data = dCval;
        matC->row_data = dCptr;
        matC->col_data = dCcol;
        matC->nnz = nnz_C;
        spgemmDescr->spgemm_reuse_flag = true;
      }
      break;
    }
      return ALPHA_SPARSE_STATUS_SUCCESS;
  }
}

alphasparseStatus_t
alphasparseSpGEMMreuse_compute(alphasparseHandle_t handle,
                               alphasparseOperation_t opA,
                               alphasparseOperation_t opB,
                               const void* alpha,
                               alphasparseSpMatDescr_t matA,
                               alphasparseSpMatDescr_t matB,
                               const void* beta,
                               alphasparseSpMatDescr_t matC,
                               alphasparseDataType computeType,
                               alphasparseSpGEMMAlg_t alg,
                               alphasparseSpGEMMDescr_t spgemmDescr)
{
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F) {
    return spgemm_reuse_template<int32_t, float>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F) {
    return spgemm_reuse_template<int32_t, double>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
    return spgemm_reuse_template<int32_t, alphaFloatComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
    return spgemm_reuse_template<int32_t, alphaDoubleComplex>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
      computeType == ALPHA_R_32F) {
    return spgemm_reuse_template<int32_t, half>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F) {
    return spgemm_reuse_template<int32_t, half2>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
#if (CUDA_ARCH >= 80)
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF) {
    return spgemm_reuse_template<int32_t, nv_bfloat16>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16BF && matC->data_type == ALPHA_C_16BF) {
    return spgemm_reuse_template<int32_t, nv_bfloat162>(
      handle, opA, opB, alpha, matA, matB, beta, matC, spgemmDescr);
  }
#endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
