#include "../../format/transpose_csr.h"
#include "../../format/coo2csr.h"
#include "alphasparse.h"
#include "spsv_csr_n_lo.h"
#include "spsv_csr_n_up.h"
#include "spsv_csr_u_lo.h"
#include "spsv_csr_u_up.h"
#include <iostream>

template<typename T, typename U>
alphasparseStatus_t
spsv_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              const void* alpha,
              alphasparseSpMatDescr_t matA,
              alphasparseDnVecDescr_t vecX,
              alphasparseDnVecDescr_t vecY,
              alphasparseSpSVDescr_t spsvDescr)
{
  if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    transpose_csr<T, U>(matA);
  }
  if (spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT &&
      spsvDescr->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER) {
    spsv_csr_u_lo<T, U>(handle,
                        (T)matA->rows,
                        (T)matA->nnz,
                        *((U*)alpha),
                        (U*)matA->val_data,
                        (T*)matA->row_data,
                        (T*)matA->col_data,
                        (U*)vecX->values,
                        (U*)vecY->values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  if (spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT &&
      spsvDescr->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER) {
    spsv_csr_n_lo<T, U>(handle,
                        (T)matA->rows,
                        (T)matA->nnz,
                        *((U*)alpha),
                        (U*)matA->val_data,
                        (T*)matA->row_data,
                        (T*)matA->col_data,
                        (U*)vecX->values,
                        (U*)vecY->values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  if (spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT &&
      spsvDescr->fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER) {
    spsv_csr_n_up<T, U>(handle,
                        (T)matA->rows,
                        (T)matA->nnz,
                        *((U*)alpha),
                        (U*)matA->val_data,
                        (T*)matA->row_data,
                        (T*)matA->col_data,
                        (U*)vecX->values,
                        (U*)vecY->values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  if (spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT &&
      spsvDescr->fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER) {
    spsv_csr_u_up<T, U>(handle,
                        (T)matA->rows,
                        (T)matA->nnz,
                        *((U*)alpha),
                        (U*)matA->val_data,
                        (T*)matA->row_data,
                        (T*)matA->col_data,
                        (U*)vecX->values,
                        (U*)vecY->values);
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
}

alphasparseStatus_t
alphasparseSpSV_solve(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      const void* alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseDnVecDescr_t vecX,
                      alphasparseDnVecDescr_t vecY,
                      alphasparseDataType computeType,
                      alphasparseSpSVAlg_t alg,
                      alphasparseSpSVDescr_t spsvDescr)
{
  if (matA->format == ALPHA_SPARSE_FORMAT_COO) {
    int m = matA->rows;
    int n = matA->cols;
    int nnz = matA->nnz;
    int* dCsrRowPtr = NULL;
    cudaMalloc((void**)&dCsrRowPtr, sizeof(int) * (m + 1));
    alphasparseXcoo2csr(matA->row_data, nnz, m, dCsrRowPtr);
    alphasparseSpMatDescr_t matA_csr;
    alphasparseCreateCsr(&matA_csr,
                         m,
                         n,
                         nnz,
                         dCsrRowPtr,
                         matA->col_data,
                         matA->val_data,
                         matA->row_type,
                         matA->col_type,
                         matA->idx_base,
                         matA->data_type);
    // alphasparse_fill_mode_t fillmode = ALPHA_SPARSE_FILL_MODE_UPPER;
    // alphasparseSpMatSetAttribute(
    //   matA_csr, ALPHASPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode));
    // // Specify Unit|Non-Unit diagonal type.
    // alphasparse_diag_type_t diagtype = ALPHA_SPARSE_DIAG_NON_UNIT;
    // alphasparseSpMatSetAttribute(
    //   matA_csr, ALPHASPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype));
    matA = matA_csr;
  }
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F) {
    return spsv_template<int32_t, float>(handle, opA, alpha, matA, vecX, vecY, spsvDescr);
  }
  // double real ; i64
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F) {
    return spsv_template<int32_t, double>(handle, opA, alpha, matA, vecX, vecY, spsvDescr);
  }
  // single complex ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F) {
    return spsv_template<int32_t, cuFloatComplex>(
      handle, opA, alpha, matA, vecX, vecY, spsvDescr);
  }
  // double complex ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F) {
    return spsv_template<int32_t, cuDoubleComplex>(
      handle, opA, alpha, matA, vecX, vecY, spsvDescr);
  }
}

alphasparseStatus_t
alphasparseSpSV_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnVecDescr_t vecX,
                           alphasparseDnVecDescr_t vecY,
                           alphasparseDataType computeType,
                           alphasparseSpSVAlg_t alg,
                           alphasparseSpSVDescr_t spsvDescr,
                           size_t* bufferSize)
{
  switch (matA->format) {
    case ALPHA_SPARSE_FORMAT_CSR: {
      *bufferSize = 4;
      break;
    }
    case ALPHA_SPARSE_FORMAT_COO: {
      switch (opA) {
        case ALPHA_SPARSE_OPERATION_NON_TRANSPOSE: {
          *bufferSize = (matA->rows + 1) * sizeof(decltype(matA->row_data));
          break;
        }
        case ALPHA_SPARSE_OPERATION_TRANSPOSE:
        case ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE: {
          *bufferSize = (matA->cols + 1) * sizeof(decltype(matA->row_data));
          break;
        }
      }
      break;
    }
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpSV_analysis(alphasparseHandle_t handle,
                         alphasparseOperation_t opA,
                         const void* alpha,
                         alphasparseSpMatDescr_t matA,
                         alphasparseDnVecDescr_t vecX,
                         alphasparseDnVecDescr_t vecY,
                         alphasparseDataType computeType,
                         alphasparseSpSVAlg_t alg,
                         alphasparseSpSVDescr_t spsvDescr,
                         void* externalBuffer)
{
  spsvDescr->diag_type = matA->descr->diag_type;
  spsvDescr->fill_mode = matA->descr->fill_mode;
  spsvDescr->type = matA->descr->type;
  spsvDescr->base = matA->descr->base;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
