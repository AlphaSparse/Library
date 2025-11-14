#include "../test_common.h"

#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <iomanip>
#include <iostream>
#include <vector>
#include <chrono>

#include "alphasparse.h"
#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"

#define CHECK_HIP(cmd)                                                                                 \
  do                                                                                                   \
  {                                                                                                    \
    hipError_t err = cmd;                                                                              \
    if (err != hipSuccess)                                                                             \
    {                                                                                                  \
      fprintf(stderr, "HIP Error: %s in %s at line %d\n", hipGetErrorString(err), __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                              \
    }                                                                                                  \
  } while (0)

#define CHECK_CUSPARSE(cmd)                                                                  \
  do                                                                                         \
  {                                                                                          \
    hipsparseStatus_t status = cmd;                                                          \
    if (status != HIPSPARSE_STATUS_SUCCESS)                                                  \
    {                                                                                        \
      fprintf(stderr, "hipSPARSE Error: %d in %s at line %d\n", status, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                                    \
    }                                                                                        \
  } while (0)

const char *file, *metrics_file;
int thread_num;
bool check_flag;
bool metrics_flag;
int iter, warm_up = 10, trials = 100;
alphasparseOperation_t transA;
int m, n, nnz;
int *coo_row_index, *coo_col_index;
double *coo_values;
double *x_val, *ict_y, *cuda_y;
const double alpha = 2.1f, beta = 3.2f;

std::vector<double> cuda_time_list, alpha_time_list, cuda_bandwidth_list, alpha_bandwidth_list, cuda_gflops_list, alpha_gflops_list;
std::vector<hipsparseSpMVAlg_t> cu_alg_list = {HIPSPARSE_MV_ALG_DEFAULT, HIPSPARSE_CSRMV_ALG1, HIPSPARSE_CSRMV_ALG2};
std::vector<alphasparseSpMVAlg_t> alpha_alg_list = {
    ALPHA_SPARSE_SPMV_ALG_VECTOR,
    ALPHA_SPARSE_SPMV_ROW_PARTITION,
    ALPHA_SPARSE_SPMV_ALG_MERGE,
    ALPHA_SPARSE_SPMV_ALG_FLAT1,
    ALPHA_SPARSE_SPMV_ALG_FLAT4,
    ALPHA_SPARSE_SPMV_ALG_FLAT8,
    ALPHA_SPARSE_SPMV_ALG_LINE_ENHANCE};

hipEvent_t event_start, event_stop;
float elapsed_time = 0.0;

static void cuda_mv()
{
  printf("\n--- Starting cuda_mv() [hipSPARSE] ---\n");
  hipsparseHandle_t handle = NULL;
  CHECK_CUSPARSE(hipsparseCreate(&handle));

  double *d_x = NULL, *d_y = NULL;
  int *d_csr_row_ptr = NULL, *d_csr_col_ind = NULL;
  double *d_csr_val = NULL;
  int *d_coo_row_ind_temp = NULL;

  CHECK_HIP(hipMalloc((void **)&d_csr_row_ptr, sizeof(int) * (m + 1)));
  CHECK_HIP(hipMalloc((void **)&d_csr_col_ind, sizeof(int) * nnz));
  CHECK_HIP(hipMalloc((void **)&d_csr_val, sizeof(double) * nnz));
  CHECK_HIP(hipMalloc((void **)&d_coo_row_ind_temp, sizeof(int) * nnz));
  CHECK_HIP(hipMalloc((void **)&d_x, n * sizeof(double)));
  CHECK_HIP(hipMalloc((void **)&d_y, m * sizeof(double)));
  CHECK_HIP(hipMemcpy(d_coo_row_ind_temp, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_csr_col_ind, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_csr_val, coo_values, nnz * sizeof(double), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_x, x_val, n * sizeof(double), hipMemcpyHostToDevice));

  alphasparseXcoo2csr(d_coo_row_ind_temp, nnz, m, d_csr_row_ptr);
  CHECK_HIP(hipFree(d_coo_row_ind_temp));
  d_coo_row_ind_temp = NULL;

  hipsparseSpMatDescr_t matA;
  hipsparseDnVecDescr_t vecX, vecY;
  CHECK_CUSPARSE(hipsparseCreateCsr(&matA, m, n, nnz, d_csr_row_ptr, d_csr_col_ind, d_csr_val, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F));
  CHECK_CUSPARSE(hipsparseCreateDnVec(&vecX, n, d_x, HIP_R_64F));
  CHECK_CUSPARSE(hipsparseCreateDnVec(&vecY, m, d_y, HIP_R_64F));

  size_t bufferSize = 0;
  void *dBuffer = NULL;

  for (auto alg : cu_alg_list)
  {
    CHECK_CUSPARSE(hipsparseSpMV_bufferSize(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, HIP_R_64F, alg, &bufferSize));
    if (bufferSize > 0)
      CHECK_HIP(hipMalloc(&dBuffer, bufferSize));
    for (int i = 0; i < warm_up; ++i)
    {
      CHECK_HIP(hipMemcpy(d_y, cuda_y, m * sizeof(double), hipMemcpyHostToDevice));
      CHECK_CUSPARSE(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, HIP_R_64F, alg, dBuffer));
    }
    hipDeviceSynchronize();
    std::vector<double> times;
    for (int i = 0; i < trials; ++i)
    {
      CHECK_HIP(hipMemcpy(d_y, cuda_y, m * sizeof(double), hipMemcpyHostToDevice));
      GPU_TIMER_START(elapsed_time, event_start, event_stop);
      CHECK_CUSPARSE(hipsparseSpMV(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, HIP_R_64F, alg, dBuffer));
      GPU_TIMER_END(elapsed_time, event_start, event_stop);
      times.push_back(elapsed_time);
    }
    hipDeviceSynchronize();
    if (dBuffer)
    {
      CHECK_HIP(hipFree(dBuffer));
      dBuffer = NULL;
    }
    double time = get_avg_time(times);
    printf("hipSPARSE alg %d: %lf ms\n", alg, time);
    cuda_time_list.push_back(time);
  }
  CHECK_HIP(hipMemcpy(cuda_y, d_y, sizeof(double) * m, hipMemcpyDeviceToHost));
  CHECK_CUSPARSE(hipsparseDestroySpMat(matA));
  CHECK_CUSPARSE(hipsparseDestroyDnVec(vecX));
  CHECK_CUSPARSE(hipsparseDestroyDnVec(vecY));
  CHECK_CUSPARSE(hipsparseDestroy(handle));
  CHECK_HIP(hipFree(d_csr_row_ptr));
  CHECK_HIP(hipFree(d_csr_col_ind));
  CHECK_HIP(hipFree(d_csr_val));
  CHECK_HIP(hipFree(d_x));
  CHECK_HIP(hipFree(d_y));
  printf("--- Finished cuda_mv() ---\n");
}

static void alpha_mv()
{
  printf("\n--- Starting alpha_mv() [alphasparse] ---\n");

  double *d_x = NULL, *d_y = NULL;
  int *d_csr_row_ptr = NULL, *d_coo_row_ind = NULL, *d_csr_col_ind = NULL;
  double *d_csr_val = NULL;

  CHECK_HIP(hipMalloc((void **)&d_coo_row_ind, sizeof(int) * nnz));
  CHECK_HIP(hipMalloc((void **)&d_csr_col_ind, sizeof(int) * nnz));
  CHECK_HIP(hipMalloc((void **)&d_csr_val, sizeof(double) * nnz));
  CHECK_HIP(hipMalloc((void **)&d_csr_row_ptr, sizeof(int) * (m + 1)));
  CHECK_HIP(hipMalloc((void **)&d_x, n * sizeof(double)));
  CHECK_HIP(hipMalloc((void **)&d_y, m * sizeof(double)));
  CHECK_HIP(hipMemcpy(d_coo_row_ind, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_csr_col_ind, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_csr_val, coo_values, nnz * sizeof(double), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_x, x_val, n * sizeof(double), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(d_coo_row_ind, nnz, m, d_csr_row_ptr);
  CHECK_HIP(hipFree(d_coo_row_ind));
  d_coo_row_ind = NULL;

  alphasparseSpMatDescr_t csrA;
  alphasparseDnVecDescr_t vecX, vecY;
  alphasparseCreateCsr(&csrA, m, n, nnz, d_csr_row_ptr, d_csr_col_ind, d_csr_val, ALPHA_SPARSE_INDEXTYPE_I32, ALPHA_SPARSE_INDEXTYPE_I32, ALPHA_SPARSE_INDEX_BASE_ZERO, ALPHA_R_64F);
  alphasparseCreateDnVec(&vecX, n, (void *)d_x, ALPHA_R_64F);
  alphasparseCreateDnVec(&vecY, m, (void *)d_y, ALPHA_R_64F);

  void *dBuffer = NULL;
  size_t bufferSize = 0;
  for (auto alg : alpha_alg_list)
  {
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    {
      alphasparseSpMV_bufferSize(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrA, vecX, &beta, vecY, ALPHA_R_64F, alg, &bufferSize);
      if (bufferSize > 0)
        CHECK_HIP(hipMalloc(&dBuffer, bufferSize));
      for (int i = 0; i < warm_up; ++i)
      {
        CHECK_HIP(hipMemcpy(d_y, ict_y, m * sizeof(double), hipMemcpyHostToDevice));
        alphasparseSpMV(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrA, vecX, &beta, vecY, ALPHA_R_64F, alg, dBuffer);
      }
      hipDeviceSynchronize();
      std::vector<double> times;
      for (int i = 0; i < trials; ++i)
      {
        CHECK_HIP(hipMemcpy(d_y, ict_y, m * sizeof(double), hipMemcpyHostToDevice));
        GPU_TIMER_START(elapsed_time, event_start, event_stop);
        alphasparseSpMV(handle, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha, csrA, vecX, &beta, vecY, ALPHA_R_64F, alg, dBuffer);
        GPU_TIMER_END(elapsed_time, event_start, event_stop);
        times.push_back(elapsed_time);
      }
      hipDeviceSynchronize();
      if (dBuffer)
      {
        CHECK_HIP(hipFree(dBuffer));
        dBuffer = NULL;
      }
      double time = get_avg_time(times);
      printf("alphasparse alg %d: %lf ms\n", alg, time);
      alpha_time_list.push_back(time);
    }
  }

  CHECK_HIP(hipMemcpy(ict_y, d_y, sizeof(double) * m, hipMemcpyDeviceToHost));
  CHECK_HIP(hipFree(d_csr_row_ptr));
  CHECK_HIP(hipFree(d_csr_col_ind));
  CHECK_HIP(hipFree(d_csr_val));
  CHECK_HIP(hipFree(d_x));
  CHECK_HIP(hipFree(d_y));
  printf("--- Finished alpha_mv() ---\n");
}

int main(int argc, const char *argv[])
{
  args_help_hip(argc, argv);
  file = args_get_data_file(argc, argv);
  metrics_file = args_save_metrics_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  metrics_flag = args_get_if_calculate_metrics(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  const char *libname = args_get_libname(argc, argv);

  alpha_read_coo<double>(file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, double>(nnz, coo_row_index, coo_col_index, coo_values);

  x_val = (double *)alpha_malloc(n * sizeof(double));
  ict_y = (double *)alpha_malloc(m * sizeof(double));
  cuda_y = (double *)alpha_malloc(m * sizeof(double));
  alpha_fill_random(x_val, 2, n);
  alpha_fill_random(ict_y, 1, m);
  memcpy(cuda_y, ict_y, m * sizeof(double));

  printf("\nMatrix: %s\n", file);
  printf("Dimensions: %d x %d, NNZ: %d\n", m, n, nnz);
  printf("Parameters: alpha=%.2f, beta=%.2f\n", alpha, beta);
  printf("Warm-up: %d, Trials: %d\n", warm_up, trials);

  cuda_mv();
  alpha_mv();

  printf("\n--- Checking results... ---\n");
  check(cuda_y, m, ict_y, m);

  alpha_free(coo_row_index);
  alpha_free(coo_col_index);
  alpha_free(coo_values);
  alpha_free(x_val);
  alpha_free(ict_y);
  alpha_free(cuda_y);

  return 0;
}