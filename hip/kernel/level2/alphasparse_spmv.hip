#include "../../format/transpose_csr.h"
#include "alphasparse.h"
// #include "alphasparse_spmv_coo.h"
#include "alphasparse_spmv_csc.h"
#include "alphasparse_spmv_csr_scalar.h"
#include "alphasparse_spmv_csr_vector.h"
#include "alphasparse_spmv_csr_row_partition.h"
#include "alphasparse_spmv_csr_merge_ginkgo.h"
#include "alphasparse_spmv_csr_merge.h"
// #include "alphasparse_spmv_csr_row_partition_ginkgo.h"
#include "alphasparse_spmv_csr_adaptive.h"
#include "alphasparse_spmv_csr_adaptive2.h"
#include "alphasparse_spmv_csr_acc.h"
// #include "alphasparse_spmv_coo_row_partition_ginkgo.h"
#include <iostream>

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t
spmv_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              const void *alpha,
              alphasparseSpMatDescr_t matA,
              alphasparseDnVecDescr_t vecX,
              const void *beta,
              alphasparseDnVecDescr_t vecY,
              alphasparseSpMVAlg_t alg,
              void *externalBuffer)
{
  switch (matA->format)
  {
  case ALPHA_SPARSE_FORMAT_COO:
  {
    break;
  }
  case ALPHA_SPARSE_FORMAT_CSR:
  {
    if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE)
    {
      transpose_csr<T, U>(matA);
    }
    switch (alg)
    {
    case ALPHA_SPARSE_SPMV_ALG_SCALAR:
    {
      spmv_csr_scalar<T, U, V, W>(handle,
                                  (T)matA->rows,
                                  (T)matA->cols,
                                  (T)matA->nnz,
                                  *((W *)alpha),
                                  (U *)matA->val_data,
                                  (T *)matA->row_data,
                                  (T *)matA->col_data,
                                  (U *)vecX->values,
                                  *((W *)beta),
                                  (V *)vecY->values);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_VECTOR:
    {
      spmv_csr_vector<T, U, V, W>(handle,
                                  (T)matA->rows,
                                  (T)matA->cols,
                                  (T)matA->nnz,
                                  *((W *)alpha),
                                  (U *)matA->val_data,
                                  (T *)matA->row_data,
                                  (T *)matA->col_data,
                                  (U *)vecX->values,
                                  *((W *)beta),
                                  (V *)vecY->values);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_MERGE:
    {
      spmv_csr_merge_ginkgo<T, U, V, W>(handle,
                                        (T)matA->rows,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W *)alpha),
                                        (U *)matA->val_data,
                                        (T *)matA->row_data,
                                        (T *)matA->col_data,
                                        (U *)vecX->values,
                                        *((W *)beta),
                                        (V *)vecY->values,
                                        externalBuffer);
      break;
    }
    case ALPHA_SPARSE_SPMV_ALG_ACC:
    {
      spmv_csr_acc<T, U, V, W>(handle,
                                        (T)matA->rows,
                                        (T)matA->cols,
                                        (T)matA->nnz,
                                        *((W *)alpha),
                                        (U *)matA->val_data,
                                        (T *)matA->row_data,
                                        (T *)matA->col_data,
                                        (U *)vecX->values,
                                        *((W *)beta),
                                        (V *)vecY->values,
                                        externalBuffer);
      break;
    }
    case ALPHA_SPARSE_SPMV_ADAPTIVE:
    {
      spmv_csr_adaptive<T, U, V, W>(handle,
                                     (T)matA->rows,
                                     (T)matA->cols,
                                     (T)matA->nnz,
                                     *((W *)alpha),
                                     (U *)matA->val_data,
                                     (T *)matA->row_data,
                                     (T *)matA->col_data,
                                     (U *)vecX->values,
                                     *((W *)beta),
                                     (V *)vecY->values);
      break;
    }
    }
    break;
  }
  case ALPHA_SPARSE_FORMAT_CSC:
  {
    spmv_csc<T, U, V, W>(handle,
                         (T)matA->rows,
                         (T)matA->cols,
                         (T)matA->nnz,
                         *((W *)alpha),
                         (U *)matA->val_data,
                         (T *)matA->row_data,
                         (T *)matA->col_data,
                         (U *)vecX->values,
                         *((W *)beta),
                         (V *)vecY->values);
    break;
  }
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpMV(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                const void *alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseDnVecDescr_t vecX,
                const void *beta,
                alphasparseDnVecDescr_t vecY,
                alphasparseDataType computeType,
                alphasparseSpMVAlg_t alg,
                void *externalBuffer)
{
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && vecY->data_type == ALPHA_R_32F)
  {
    return spmv_template<int32_t, float, float, float>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && vecY->data_type == ALPHA_R_64F)
  {
    return spmv_template<int32_t, double, double, double>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && vecY->data_type == ALPHA_R_32I)
  {
    return spmv_template<int32_t, int8_t, int32_t, int32_t>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_8I && vecY->data_type == ALPHA_R_32F)
  {
    return spmv_template<int32_t, int8_t, float, float>(
        handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_32F &&
  //     vecY->data_type == ALPHA_R_32F) {
  //   return spmv_template<int32_t, float, float>(
  //     handle, opA, alpha, matA, vecX, beta, vecY, alg, externalBuffer);
  // }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpMV_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           const void *alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnVecDescr_t vecX,
                           const void *beta,
                           alphasparseDnVecDescr_t vecY,
                           alphasparseDataType computeType,
                           alphasparseSpMVAlg_t alg,
                           size_t *bufferSize)
{
  switch (alg)
  {
  case ALPHA_SPARSE_SPMV_ALG_MERGE:
  {
    const int total = matA->rows + matA->nnz;
    const int block_num =
        ceildivT(total, SPMV_MERGE_BLOCK_SIZE * 8);
    size_t typeSize = 4;
    if (computeType == ALPHA_R_32F)
      typeSize = 4;
    else if (computeType == ALPHA_R_64F)
      typeSize = 8;
    else if (computeType == ALPHA_C_32F)
      typeSize = 8;
    else if (computeType == ALPHA_C_64F)
      typeSize = 16;
    *bufferSize = (block_num * 2 + 2) * typeSize;
    break;
  }
  case ALPHA_SPARSE_SPMV_ALG_ACC:
  {
    constexpr int R = 2;
    const int BLOCK_SIZE = 512;
    const int nnz_per_block = R * BLOCK_SIZE;
    const int nwarps = ceildivT<int>(matA->nnz, nnz_per_block);
    *bufferSize = (nwarps + 1) * sizeof(int);
    break;
  }
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
