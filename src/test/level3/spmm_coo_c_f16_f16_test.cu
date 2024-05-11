
#include "../test_common.h"

/**
 * @brief ict dcu mv hyb test
 * @author HPCRC, ICT
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_coo.h"
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
half2* coo_values;

// parms for kernel
half2 *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
long long B_rows, B_cols;
long long ldb, ldc;
long long B_size, C_size;
const cuFloatComplex alpha = {2.f,3.f};
const cuFloatComplex beta = {3.f,2.f};

#define CHECK_CUDA(func)                                                       \
  {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
      printf("CUDA API failed at line %d with error: %s (%d)\n",               \
             __LINE__,                                                         \
             cudaGetErrorString(status),                                       \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

#define CHECK_CUSPARSE(func)                                                   \
  {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
             __LINE__,                                                         \
             cusparseGetErrorString(status),                                   \
             status);                                                          \
      exit(-1);                                                                \
    }                                                                          \
  }

static void
cuda_mm()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  // Offload data to device
  int* dArow = NULL;
  int* dAcol = NULL;
  half2* dAval = NULL;

  int nnz = rnnz;

  half2* dmatB = NULL;
  half2* dmatC = NULL;

  cudaMalloc((void**)&dmatB, sizeof(half2) * B_size);
  cudaMalloc((void**)&dmatC, sizeof(half2) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(half2) * nnz));

  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(half2), cudaMemcpyHostToDevice));

  cudaMemcpy(dmatB, hmatB, sizeof(half2) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_roc, sizeof(half2) * C_size, cudaMemcpyHostToDevice);
  cusparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(cusparseCreateDnMat(
    &matB, A_cols, B_cols, ldb, dmatB, CUDA_C_16F, CUSPARSE_ORDER_COL))
  // Create dense matrix C
  CHECK_CUSPARSE(cusparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, CUDA_C_16F, CUSPARSE_ORDER_COL))
  cusparseSpMatDescr_t matA;
  CHECK_CUSPARSE(cusparseCreateCoo(&matA,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dArow,
                                   dAcol,
                                   dAval,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_C_16F));
  size_t bufferSize = 0;
  cusparseOperation_t cutransA, cutransB;
  if(transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) cutransA = CUSPARSE_OPERATION_TRANSPOSE;
  else if(transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) cutransA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else cutransA = CUSPARSE_OPERATION_NON_TRANSPOSE;

  if(transB == ALPHA_SPARSE_OPERATION_TRANSPOSE) cutransB = CUSPARSE_OPERATION_TRANSPOSE;
  else if(transB == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) cutransB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  else cutransB = CUSPARSE_OPERATION_NON_TRANSPOSE;

  CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                         cutransA,
                                         cutransB,
                                         &alpha,
                                         matA,
                                         matB,
                                         &beta,
                                         matC,
                                         CUDA_C_32F,
                                         CUSPARSE_SPMM_ALG_DEFAULT,
                                         &bufferSize))
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  printf(" CUDA TRANS %d\n",cutransA);
  CHECK_CUSPARSE(cusparseSpMM(handle,
                              cutransA,
                              cutransB,
                              &alpha,
                              matA,
                              matB,
                              &beta,
                              matC,
                              CUDA_C_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT,
                              dBuffer))
  CHECK_CUDA(
    cudaMemcpy(matC_roc, dmatC, C_size * sizeof(half2), cudaMemcpyDeviceToHost))
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dmatB);
  cudaFree(dmatC);
  cusparseDestroy(handle);
}

static void
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  int* dArow = NULL;
  int* dAcol = NULL;
  half2* dAval = NULL;

  int nnz = rnnz;

  half2* dmatB = NULL;
  half2* dmatC = NULL;

  cudaMalloc((void**)&dmatB, sizeof(half2) * B_size);
  cudaMalloc((void**)&dmatC, sizeof(half2) * C_size);

  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_CUDA_ERROR(cudaMalloc((void**)&dAval, sizeof(half2) * nnz));

  CHECK_CUDA(cudaMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(dAval, coo_values, nnz * sizeof(half2), cudaMemcpyHostToDevice));

  cudaMemcpy(dmatB, hmatB, sizeof(half2) * B_size, cudaMemcpyHostToDevice);
  cudaMemcpy(dmatC, matC_ict, sizeof(half2) * C_size, cudaMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
    &matB, A_cols, B_cols, ldb, dmatB, ALPHA_C_16F, ALPHASPARSE_ORDER_COL);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_C_16F, ALPHASPARSE_ORDER_COL);
  alphasparseSpMatDescr_t matA;
  alphasparseCreateCoo(&matA,
                       A_rows,
                       A_cols,
                       nnz,
                       dArow,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_C_16F);
  size_t bufferSize = 0;
  alphasparseSpMM_bufferSize(handle,
                             transA,
                             transB,
                             &alpha,
                             matA,
                             matB,
                             &beta,
                             matC,
                             ALPHA_C_32F,
                             ALPHASPARSE_SPMM_ALG_DEFAULT,
                             &bufferSize);
  void* dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  printf(" ALPHA TRANS %d\n", transA);
  alphasparseSpMM(handle,
                  transA,
                  transB,
                  &alpha,
                  matA,
                  matB,
                  &beta,
                  matC,
                  ALPHA_C_32F,
                  ALPHASPARSE_SPMM_ALG_DEFAULT,
                  dBuffer);
  CHECK_CUDA(
    cudaMemcpy(matC_ict, dmatC, C_size * sizeof(half2), cudaMemcpyDeviceToHost))
  // Clear up on device
  cudaFree(dArow);
  cudaFree(dAcol);
  cudaFree(dAval);
  cudaFree(dmatB);
  cudaFree(dmatC);
}

int
main(int argc, const char* argv[])
{
  args_help(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);

  // read coo
  alpha_read_coo<half2>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, half2>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, A_cols); // 默认C是方阵
  
  C_cols = columns;
  B_cols = columns;
  C_rows = A_rows;
  B_rows = A_cols;
  
  ldb = columns;
  ldc = columns;
  B_size = ldb * B_rows;
  C_size = ldc * B_rows;
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
  hmatB = (half2*)alpha_malloc(B_size * sizeof(half2));
  matC_ict = (half2*)alpha_malloc(C_size * sizeof(half2));
  matC_roc = (half2*)alpha_malloc(C_size * sizeof(half2));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  alpha_fill_random(matC_roc, 1, C_size);

  cuda_mm();
  alpha_mm();

  for (int i = 0; i < 40; i++) {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 40; i++) {
    std::cout << matC_ict[i] << ", ";
  }
  std::cout << std::endl;
  check((half2*)matC_roc, C_size, (half2*)matC_ict, C_size);
  return 0;
}
