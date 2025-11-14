
#include "../test_common.h"

/**
 * @brief ict dcu mv hyb test
 * @author HPCRC, ICT
 */

#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
int8_t* coo_values;

// parms for kernel
int8_t *hmatB;
int32_t *matC_ict, *matC_roc;
long long C_rows, C_cols;
long long B_cols;
long long ldb, ldc;
long long B_size, C_size;
const int32_t alpha = 2;
const int32_t beta = 3;

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    hipError_t status = (func);                                               \
    if (status != hipSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             hipGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    hipError_t status = (func);                                          \
    if (status != HIPSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
             __LINE__,                                                         \
             hipGetErrorString(status),                                   \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

static void
cuda_mm()
{
  hipsparseHandle_t handle = NULL;
  CHECK_CUSPARSE(hipsparseCreate(&handle));

  // Offload data to device
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  int8_t* dAval = NULL;

  int nnz = rnnz;

  int8_t* dmatB = NULL;
  int32_t* dmatC = NULL;

  hipMalloc((void**)&dmatB, sizeof(int8_t) * B_size);
  hipMalloc((void**)&dmatC, sizeof(int32_t) * C_size);

  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(int8_t) * nnz));
  PRINT_IF_HIP_ERROR(
    hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
    hipMemcpy(dAval, coo_values, nnz * sizeof(int8_t), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  hipMemcpy(dmatB, hmatB, sizeof(int8_t) * B_size, hipMemcpyHostToDevice);
  hipMemcpy(dmatC, matC_roc, sizeof(int32_t) * C_size, hipMemcpyHostToDevice);
  hipsparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(hipsparseCreateDnMat(
    &matB, A_cols, B_cols, ldb, dmatB, HIP_R_8I, HIPSPARSE_ORDER_COL))
  // Create dense matrix C
  CHECK_CUSPARSE(hipsparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, HIP_R_32I, HIPSPARSE_ORDER_COL))
  hipsparseSpMatDescr_t matA;
  CHECK_CUSPARSE(hipsparseCreateCsr(&matA,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dCsrRowPtr,
                                   dAcol,
                                   dAval,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_BASE_ZERO,
                                   HIP_R_8I));
  size_t bufferSize = 0;
  CHECK_CUSPARSE(hipsparseSpMM_bufferSize(handle,
                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matB,
                                         &beta,
                                         matC,
                                         HIP_R_32I,
                                         HIPSPARSE_SPMM_ALG_DEFAULT,
                                         &bufferSize))
  void* dBuffer = NULL;
  CHECK_CUDA(hipMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(hipsparseSpMM(handle,
                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              matA,
                              matB,
                              &beta,
                              matC,
                              HIP_R_32I,
                              HIPSPARSE_SPMM_ALG_DEFAULT,
                              dBuffer))
  CHECK_CUDA(
    hipMemcpy(matC_roc, dmatC, C_size * sizeof(int32_t), hipMemcpyDeviceToHost))
  // Clear up on device
  hipFree(dArow);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dmatB);
  hipFree(dmatC);
  hipsparseDestroy(handle);
}

static void
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  int8_t* dAval = NULL;

  int nnz = rnnz;

  int8_t* dmatB = NULL;
  int32_t* dmatC = NULL;

  hipMalloc((void**)&dmatB, sizeof(int8_t) * B_size);
  hipMalloc((void**)&dmatC, sizeof(int32_t) * C_size);

  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(int8_t) * nnz));
  PRINT_IF_HIP_ERROR(
    hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
    hipMemcpy(dAval, coo_values, nnz * sizeof(int8_t), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  hipMemcpy(dmatB, hmatB, sizeof(int8_t) * B_size, hipMemcpyHostToDevice);
  hipMemcpy(dmatC, matC_ict, sizeof(int32_t) * C_size, hipMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
    &matB, A_cols, B_cols, ldb, dmatB, ALPHA_R_8I, ALPHASPARSE_ORDER_COL);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_R_32I, ALPHASPARSE_ORDER_COL);
  alphasparseSpMatDescr_t matA;
  alphasparseCreateCsr(&matA,
                       A_rows,
                       A_cols,
                       nnz,
                       dCsrRowPtr,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_8I);
  size_t bufferSize = 0;
  alphasparseSpMM_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             matA,
                             matB,
                             &beta,
                             matC,
                             ALPHA_R_32I,
                             ALPHASPARSE_SPMM_ALG_DEFAULT,
                             &bufferSize);
  void* dBuffer = NULL;
  CHECK_CUDA(hipMalloc(&dBuffer, bufferSize));
  alphasparseSpMM(handle,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                  &alpha,
                  matA,
                  matB,
                  &beta,
                  matC,
                  ALPHA_R_32I,
                  ALPHASPARSE_SPMM_ALG_DEFAULT,
                  dBuffer);
  CHECK_CUDA(
    hipMemcpy(matC_ict, dmatC, C_size * sizeof(int32_t), hipMemcpyDeviceToHost))
  // Clear up on device
  hipFree(dArow);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dmatB);
  hipFree(dmatC);
}

int
main(int argc, const char* argv[])
{
  args_help_hip(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);

  // read coo
  alpha_read_coo<int8_t>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, int8_t>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, A_rows); // 默认C是方阵
  C_rows = A_rows;
  C_cols = columns;
  B_cols = columns;
  ldb = A_cols;
  ldc = C_rows;
  B_size = ldb * B_cols;
  C_size = ldc * B_cols;
  for (int i = 0; i < 20; i++) {
    std::cout << coo_row_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << coo_col_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << coo_values[i] << ", ";
  }
  std::cout << std::endl;
  // init x y
  // init B C
  hmatB = (int8_t*)alpha_malloc(B_size * sizeof(int8_t));
  matC_ict = (int32_t*)alpha_malloc(C_size * sizeof(int32_t));
  matC_roc = (int32_t*)alpha_malloc(C_size * sizeof(int32_t));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  alpha_fill_random(matC_roc, 1, C_size);

  cuda_mm();
  alpha_mm();

  for (int i = 0; i < 20; i++) {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++) {
    std::cout << matC_ict[i] << ", ";
  }
  check((int32_t*)matC_roc, C_size, (int32_t*)matC_ict, C_size);
  return 0;
}
