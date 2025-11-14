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

double cu_time;
std::vector<hipsparseSpMMAlg_t> cu_alg_list = {HIPSPARSE_SPMM_ALG_DEFAULT, HIPSPARSE_SPMM_CSR_ALG1, HIPSPARSE_SPMM_CSR_ALG2, HIPSPARSE_SPMM_CSR_ALG3};
// std::vector<hipsparseSpMMAlg_t> cu_alg_list = {HIPSPARSE_SPMM_CSR_ALG2};
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
cuda_mm()
{
  hipsparseHandle_t handle = NULL;
  CHECK_CUSPARSE(hipsparseCreate(&handle));

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
  hipMemcpy(dmatC, matC_roc, sizeof(hipFloatComplex) * C_size, hipMemcpyHostToDevice);
  hipsparseDnMatDescr_t matB, matC;
  CHECK_CUSPARSE(hipsparseCreateDnMat(
    &matB, A_cols, C_cols, ldb, dmatB, HIP_C_32F, HIPSPARSE_ORDER_ROW))
  // Create dense matrix C
  CHECK_CUSPARSE(hipsparseCreateDnMat(
    &matC, C_rows, C_cols, ldc, dmatC, HIP_C_32F, HIPSPARSE_ORDER_ROW))
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
                                   HIP_C_32F));
  auto alg = cu_alg_list[alg_num];
  size_t bufferSize = 0;
  CHECK_CUSPARSE(hipsparseSpMM_bufferSize(handle,
                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                         HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matB,
                                         &beta,
                                         matC,
                                         HIP_C_32F,
                                         alg,
                                         &bufferSize))
  void* dBuffer = NULL;
  CHECK_CUDA(hipMalloc(&dBuffer, bufferSize));
  std::vector<double> times;
  for (int i = 0; i < iteration; i++) {
    if (i >= warmup_times) hipMemcpy(dmatC, matC_roc, sizeof(hipFloatComplex) * C_size, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    CHECK_CUSPARSE(hipsparseSpMM(handle,
                                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha,
                                matA,
                                matB,
                                &beta,
                                matC,
                                HIP_C_32F,
                                alg,
                                dBuffer))
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    if (i >= warmup_times) times.push_back(elapsed_time);
  }
  cu_time = get_avg_time_2(times);
  printf("cusparse %d: %lf ms\n", alg, cu_time);

  CHECK_CUDA(
  hipMemcpy(matC_roc, dmatC, C_size * sizeof(hipFloatComplex), hipMemcpyDeviceToHost))
  // Clear up on device
  hipFree(dArow);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dmatB);
  hipFree(dmatC);
  hipsparseDestroy(handle);
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
  ldb = columns;
  ldc = C_cols;
  B_size = ldb * B_rows;
  C_size = ldc * C_rows;
  // init x y
  // init B C
  hmatB = (hipFloatComplex*)alpha_malloc(B_size * sizeof(hipFloatComplex));
  matC_roc = (hipFloatComplex*)alpha_malloc(C_size * sizeof(hipFloatComplex));

  alpha_fill_random(hmatB, 0, B_size);
  alpha_fill_random(matC_roc, 1, C_size);
  cuda_mm();

  std::ofstream filename(metric_file, std::ios::app);
  filename << file << "," << cu_time << "\n";
  filename.close();

  return 0;
}
