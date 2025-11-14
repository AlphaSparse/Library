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

const int iteration = 210;
const int warmup_times = 200;
const char *file, *metric_file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

double alpha_time;
std::vector<alphasparseSpMMAlg_t> alpha_alg_list = {ALPHASPARSE_SPMM_ALG_DEFAULT, ALPHASPARSE_SPMM_CSR_ALG1, ALPHASPARSE_SPMM_CSR_ALG2, ALPHASPARSE_SPMM_CSR_ALG3, ALPHASPARSE_SPMM_CSR_ALG4, ALPHASPARSE_SPMM_CSR_ALG5};
// std::vector<alphasparseSpMMAlg_t> alpha_alg_list = {ALPHASPARSE_SPMM_CSR_ALG2};
int alg_num;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
hipFloatComplex* coo_values;

// parms for kernel
hipFloatComplex *hmatB, *matC_ict, *matC_roc;
long long C_rows, C_cols;
int B_rows;
long long ldb, ldc;
long long B_size, C_size;
const hipFloatComplex alpha = {2.f, 3.f};
const hipFloatComplex beta = {3.f, 2.f};

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
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  // Offload data to device
  int* dCsrRowPtr = NULL;
  int* dArow = NULL;
  int* dAcol = NULL;
  hipFloatComplex* dAval = NULL;

  int nnz = rnnz;

  hipFloatComplex* dmatB = NULL;
  hipFloatComplex* dmatC = NULL;

  hipMalloc((void**)&dmatB, sizeof(hipFloatComplex) * B_size);
  hipMalloc((void**)&dmatC, sizeof(hipFloatComplex) * C_size);

  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(hipFloatComplex) * nnz));
  PRINT_IF_HIP_ERROR(
    hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
    dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
    dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
    hipMemcpy(dAval, coo_values, nnz * sizeof(hipFloatComplex), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);

  hipMemcpy(dmatB, hmatB, sizeof(hipFloatComplex) * B_size, hipMemcpyHostToDevice);
  hipMemcpy(dmatC, matC_ict, sizeof(hipFloatComplex) * C_size, hipMemcpyHostToDevice);
  alphasparseDnMatDescr_t matB, matC;
  alphasparseCreateDnMat(
    &matB, A_cols, C_cols, ldb, dmatB, ALPHA_C_32F, ALPHASPARSE_ORDER_COL);
  // Create dense matrix C
  alphasparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, ALPHA_C_32F, ALPHASPARSE_ORDER_COL);
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
                       ALPHA_C_32F);
  auto alg = alpha_alg_list[alg_num];
  size_t bufferSize = 0;
  alphasparseSpMM_bufferSize(handle,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha,
                             matA,
                             matB,
                             &beta,
                             matC,
                             ALPHA_C_32F,
                             alg,
                             &bufferSize);
  void* dBuffer = NULL;
  CHECK_CUDA(hipMalloc(&dBuffer, bufferSize));
  std::vector<double> times;
  for (int i = 0; i < iteration; i++) {
    if (i >= warmup_times) hipMemcpy(dmatC, matC_ict, sizeof(hipFloatComplex) * C_size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    alphasparseSpMM(handle,
                    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                    &alpha,
                    matA,
                    matB,
                    &beta,
                    matC,
                    ALPHA_C_32F,
                    alg,
                    dBuffer);
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    if (i >= warmup_times)
      times.push_back(elapsed_time);
  }
  alpha_time = get_avg_time_2(times);
  printf("alphasparse %d: %lf ms\n", alg, alpha_time);

  CHECK_CUDA(
    hipMemcpy(matC_ict, dmatC, C_size * sizeof(hipFloatComplex), hipMemcpyDeviceToHost))
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
	metric_file = args_save_metrics_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  alg_num = args_get_alg_num(argc, argv);

  // read coo
  alpha_read_coo<hipFloatComplex>(
    file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, hipFloatComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, 256);
  C_rows = A_rows;
  C_cols = columns;
  B_rows = A_cols;
  ldb = B_rows;
  ldc = C_rows;
  B_size = ldb * C_cols;
  C_size = ldc * C_cols;
  // init x y
  // init B C
  hmatB = (hipFloatComplex*)alpha_malloc(B_size * sizeof(hipFloatComplex));
  matC_ict = (hipFloatComplex*)alpha_malloc(C_size * sizeof(hipFloatComplex));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_ict, 1, C_size);
  // std::cout << "matc:" <<std::endl;
  // for (int i = 0; i < 20; i++) {
  //   std::cout << matC_ict[i] << ", ";
  // }
  // std::cout << "matc:" << std::endl;
  alpha_mm();

  std::ofstream filename(metric_file, std::ios::app);
  filename << file << "," << alpha_time << "\n";
  filename.close();
  return 0;
}
