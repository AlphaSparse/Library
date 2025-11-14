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

hipsparseSpMMAlg_t cu_alg = HIPSPARSE_SPMM_ALG_DEFAULT;
alphasparseSpMMAlg_t alpha_alg = ALPHASPARSE_SPMM_CSR_ALG1;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
hipDoubleComplex* coo_values;

// parms for kernel
hipDoubleComplex *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
int B_rows;
long long ldb, ldc;
long long B_size, C_size;
const hipDoubleComplex alpha = {2.f, 3.f};
const hipDoubleComplex beta = {3.f, 2.f};

hipEvent_t event_start, event_stop;
float elapsed_time = 0.0;

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

#define CHECK_CUSPARSE(func) func;                                                  \
//   {                                                                            \
//     hipError_t status = (func);                                          \
//     if (status != HIPSPARSE_STATUS_SUCCESS) {                                   \
//       printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
//              __LINE__,                                                         \
//              hipGetErrorString(status),                                   \
//              status);                                                          \
//       exit(-1);                                                                \
//     }                                                                          \
//   }

static void
cuda_mm()
{
  hipsparseHandle_t handle = NULL;
  CHECK_CUSPARSE(hipsparseCreate(&handle));

  // Offload data to device
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  hipDoubleComplex* dAval = NULL;

  int nnz = rnnz;

  hipDoubleComplex* dmatB = NULL;
  hipDoubleComplex* dmatC = NULL;

  hipMalloc((void**)&dmatB, sizeof(hipDoubleComplex) * B_size);
  hipMalloc((void**)&dmatC, sizeof(hipDoubleComplex) * C_size);

  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(hipDoubleComplex) * nnz));
  PRINT_IF_HIP_ERROR(
    hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
    hipMemcpy(dAval, coo_values, nnz * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  hipMemcpy(dmatB, hmatB, sizeof(hipDoubleComplex) * B_size, hipMemcpyHostToDevice);
  hipMemcpy(dmatC, matC_roc, sizeof(hipDoubleComplex) * C_size, hipMemcpyHostToDevice);
  hipsparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(hipsparseCreateDnMat(
    &matB, A_cols, C_cols, ldb, dmatB, HIP_C_64F, HIPSPARSE_ORDER_ROW))
  // Create dense matrix C
  CHECK_CUSPARSE(hipsparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, HIP_C_64F, HIPSPARSE_ORDER_ROW))
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
                                   HIP_C_64F));
  std::vector<double> times;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  size_t bufferSize = 0;
  CHECK_CUSPARSE(hipsparseSpMM_bufferSize(handle,
                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matB,
                                         &beta,
                                         matC,
                                         HIP_C_64F,
                                         cu_alg,
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
                              HIP_C_64F,
                              cu_alg,
                              dBuffer))
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  times.push_back(elapsed_time);
  printf("cusparse: %lf\n", get_avg_time(times));
  CHECK_CUDA(
    hipMemcpy(matC_roc, dmatC, C_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost))
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
  hipDoubleComplex* dAval = NULL;

  int nnz = rnnz;

  hipDoubleComplex* dmatB = NULL;
  hipDoubleComplex* dmatC = NULL;

  hipMalloc((void**)&dmatB, sizeof(hipDoubleComplex) * B_size);
  hipMalloc((void**)&dmatC, sizeof(hipDoubleComplex) * C_size);

  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(hipDoubleComplex) * nnz));
  PRINT_IF_HIP_ERROR(
    hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
    hipMemcpy(dAval, coo_values, nnz * sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  hipMemcpy(dmatB, hmatB, sizeof(hipDoubleComplex) * B_size, hipMemcpyHostToDevice);
  hipMemcpy(dmatC, matC_ict, sizeof(hipDoubleComplex) * C_size, hipMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
    &matB, A_cols, C_cols, ldb, dmatB, ALPHA_C_64F, ALPHASPARSE_ORDER_ROW);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_C_64F, ALPHASPARSE_ORDER_ROW);
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
                       ALPHA_C_64F);
  std::vector<double> times;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  size_t bufferSize = 0;
  alphasparseSpMM_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             matA,
                             matB,
                             &beta,
                             matC,
                             ALPHA_C_64F,
                             alpha_alg,
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
                  ALPHA_C_64F,
                  alpha_alg,
                  dBuffer);
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  times.push_back(elapsed_time);
  printf("alphasparse: %lf\n", get_avg_time(times));
  CHECK_CUDA(
    hipMemcpy(matC_ict, dmatC, C_size * sizeof(hipDoubleComplex), hipMemcpyDeviceToHost))
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

  // read coo
  alpha_read_coo<hipDoubleComplex>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, hipDoubleComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, 256);
  C_rows = A_rows;
  C_cols = columns;
  B_rows = A_cols;
  ldb = columns;
  ldc = C_cols;
  B_size = ldb * B_rows;
  C_size = ldc * C_rows;
  // init x y
  // init B C
  hmatB = (hipDoubleComplex*)alpha_malloc(B_size * sizeof(hipDoubleComplex));
  matC_ict = (hipDoubleComplex*)alpha_malloc(C_size * sizeof(hipDoubleComplex));
  matC_roc = (hipDoubleComplex*)alpha_malloc(C_size * sizeof(hipDoubleComplex));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  alpha_fill_random(matC_roc, 1, C_size);
  // std::cout << "matc:" <<std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << matC_ict[i] << ", ";
  // }
  // std::cout << "matc:" << std::endl;
  cuda_mm();
  alpha_mm();

  for (int i = 0; i < 8; i++) {
    std::cout << matC_roc[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 8; i++) {
    std::cout << matC_ict[i] << ", ";
  }
  check((hipDoubleComplex*)matC_roc, C_size, (hipDoubleComplex*)matC_ict, C_size);
  return 0;
}
