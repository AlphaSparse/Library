
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

const char *file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transA;
alphasparseOperation_t transB;

long long columns;
int A_rows, A_cols, rnnz;
int *coo_row_index, *coo_col_index;
float *coo_values;

// parms for kernel
float *hmatB, *matC_ict, *matC_cuda;
long long ldb, ldc;
long long B_size, nnz, nnz_c;
const float alpha = 1.f;
const float beta = 3.f;

int trials = 10, warm_up = 5;

static void
cuda_mm()
{
  hipsparseHandle_t handle = NULL;
  CHECK_CUSPARSE(hipsparseCreate(&handle));

  int nnz = rnnz;

  // Offload data to device
  int *dACsrRowPtr = NULL;
  int *dArow = NULL;
  int *dAcol = NULL;
  float *dAval = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(
      hipMalloc((void **)&dACsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
      hipMemcpy(dAval, coo_values, nnz * sizeof(float), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dACsrRowPtr);

  hipsparseSpMatDescr_t matA;
  CHECK_CUSPARSE(hipsparseCreateCsr(&matA,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dACsrRowPtr,
                                   dAcol,
                                   dAval,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_BASE_ZERO,
                                   HIP_R_32F));

  int *dBCsrRowPtr = NULL;
  int *dBrow = NULL;
  int *dBcol = NULL;
  float *dBval = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBrow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(
      hipMalloc((void **)&dBCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
      dBrow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
      dBcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
      hipMemcpy(dBval, coo_values, nnz * sizeof(float), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dBrow, nnz, A_rows, dBCsrRowPtr);

  hipsparseSpMatDescr_t matB;
  CHECK_CUSPARSE(hipsparseCreateCsr(&matB,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dBCsrRowPtr,
                                   dBcol,
                                   dBval,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_BASE_ZERO,
                                   HIP_R_32F));

  int *dCCsrRowPtr = NULL;
  int *dCrow = NULL;
  int *dCcol = NULL;
  float *dCval = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dCrow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dCcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dCval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(
      hipMalloc((void **)&dCCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
      dCrow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
      dCcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
      hipMemcpy(dCval, coo_values, nnz * sizeof(float), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dCrow, nnz, A_rows, dCCsrRowPtr);

  hipsparseSpMatDescr_t matC;
  CHECK_CUSPARSE(hipsparseCreateCsr(&matC,
                                   A_rows,
                                   A_cols,
                                   nnz,
                                   dCCsrRowPtr,
                                   dCcol,
                                   dCval,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_32I,
                                   HIPSPARSE_INDEX_BASE_ZERO,
                                   HIP_R_32F));

  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  void *dBuffer1 = NULL;
  void *dBuffer2 = NULL;
  // SpGEMM Computation
  hipsparseSpGEMMDescr_t spgemmDesc;
  CHECK_CUSPARSE(hipsparseSpGEMM_createDescr(&spgemmDesc))
  hipEvent_t event_start, event_stop;
  float elapsed_time = 0.0;
  float elapsed_time_work = 0.0;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  // ask bufferSize1 bytes for external memory
  CHECK_CUSPARSE(hipsparseSpGEMM_workEstimation(handle,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               HIP_R_32F,
                                               HIPSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc,
                                               &bufferSize1,
                                               NULL))
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  std::cout << "hipsparseSpGEMM_workEstimation1 time: " << elapsed_time << " microseconds" << std::endl;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  CHECK_CUDA(hipMalloc((void**)&dBuffer1, bufferSize1))
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  std::cout << "hipsparseSpGEMM_workEstimation1 buffer1 time: " << elapsed_time << " microseconds" << std::endl;
  elapsed_time_work += elapsed_time;
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  CHECK_CUSPARSE(hipsparseSpGEMM_workEstimation(handle,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               matB,
                                               &beta,
                                               matC,
                                               HIP_R_32F,
                                               HIPSPARSE_SPGEMM_DEFAULT,
                                               spgemmDesc,
                                               &bufferSize1,
                                               dBuffer1))
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  elapsed_time_work += elapsed_time;
  std::cout << "hipsparseSpGEMM_workEstimation2 time: " << elapsed_time << " microseconds" << std::endl;
  // ask bufferSize2 bytes for external memory
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  CHECK_CUSPARSE(hipsparseSpGEMM_compute(handle,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        matB,
                                        &beta,
                                        matC,
                                        HIP_R_32F,
                                        HIPSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize2,
                                        NULL))
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  std::cout << "hipsparseSpGEMM_compute1 time: " << elapsed_time << " microseconds, buffersize :" << bufferSize2 << std::endl;

  double time1 = get_time_us();
  CHECK_CUDA(hipMalloc((void**)&dBuffer2, bufferSize2))
  double time2 = get_time_us();
  std::cout << "hipsparseSpGEMM buffer malloc time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
  elapsed_time_work += (time2 - time1) / (1e3);

  // compute the intermediate product of A * B
  CHECK_CUSPARSE(hipsparseSpGEMM_compute(handle,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        matB,
                                        &beta,
                                        matC,
                                        HIP_R_32F,
                                        HIPSPARSE_SPGEMM_DEFAULT,
                                        spgemmDesc,
                                        &bufferSize2,
                                        dBuffer2))
  // get matrix C non-zero entries C_nnz1
  int64_t C_num_rows1, C_num_cols1, C_nnz1;
  CHECK_CUSPARSE(
    hipsparseSpMatGetSize(matC, &C_num_rows1, &C_num_cols1, &C_nnz1))
  printf("base nnz %d\n", C_nnz1);
  nnz_c = C_nnz1;
  // allocate matrix C
  int *dC_csrOffsets, *dC_columns;
  double* dC_values;
  CHECK_CUDA(hipMalloc((void**)&dC_columns, C_nnz1 * sizeof(int)))
  CHECK_CUDA(hipMalloc((void**)&dC_values, C_nnz1 * sizeof(double)))
  CHECK_CUDA(hipMalloc((void**)&dC_csrOffsets, (A_rows + 1) * sizeof(int)))
  // NOTE: if 'beta' != 0, the values of C must be update after the allocation
  //       of dC_values, and before the call of hipsparseSpGEMM_copy

  // update matC with the new pointers
  CHECK_CUSPARSE(
    hipsparseCsrSetPointers(matC, dC_csrOffsets, dC_columns, dC_values))

  // if beta != 0, hipsparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C
  CHECK_CUSPARSE(hipsparseSpGEMM_copy(handle,
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                     HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                     &alpha,
                                     matA,
                                     matB,
                                     &beta,
                                     matC,
                                     HIP_R_32F,
                                     HIPSPARSE_SPGEMM_DEFAULT,
                                     spgemmDesc))
  // device result check
  matC_cuda = (float*)alpha_malloc(C_nnz1 * sizeof(float));
  // std::cout << "C_nnz1: " << C_nnz1 << std::endl;
  CHECK_CUDA(hipMemcpy(
    matC_cuda, dC_values, C_nnz1 * sizeof(float), hipMemcpyDeviceToHost))

  // compute the intermediate product of A * B
  for (int i = 0; i < warm_up; i++)
  {
    CHECK_CUSPARSE(hipsparseSpGEMM_compute(handle,
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          matA,
                                          matB,
                                          &beta,
                                          matC,
                                          HIP_R_32F,
                                          HIPSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize2,
                                          dBuffer2))
  }

  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  for (int i = 0; i < trials; i++)
  {
    CHECK_CUSPARSE(hipsparseSpGEMM_compute(handle,
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                          HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha,
                                          matA,
                                          matB,
                                          &beta,
                                          matC,
                                          HIP_R_32F,
                                          HIPSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc,
                                          &bufferSize2,
                                          dBuffer2))
  }
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  std::cout << "hipsparseSpGEMM_compute2 time: " << elapsed_time/trials<< " microseconds" << std::endl;
  std::cout << "hipsparseSpGEMM_compute_total time: " << elapsed_time/trials + elapsed_time_work<< " microseconds" << std::endl;

  CHECK_CUSPARSE(hipsparseSpGEMM_destroyDescr(spgemmDesc))
  CHECK_CUSPARSE(hipsparseDestroySpMat(matA))
  CHECK_CUSPARSE(hipsparseDestroySpMat(matB))
  CHECK_CUSPARSE(hipsparseDestroySpMat(matC))
  CHECK_CUSPARSE(hipsparseDestroy(handle))
  // Clear up on device
  hipFree(dArow);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dBrow);
  hipFree(dBcol);
  hipFree(dBval);
  hipFree(dCrow);
  hipFree(dCcol);
  hipFree(dCval);
}

static void
alpha_mm()
{
  alphasparseHandle_t handle = NULL;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  int nnz = rnnz;

  // Offload data to device
  int *dACsrRowPtr = NULL;
  int *dArow = NULL;
  int *dAcol = NULL;
  float *dAval = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dArow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(
      hipMalloc((void **)&dACsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
      dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
      dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
      hipMemcpy(dAval, coo_values, nnz * sizeof(float), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dArow, nnz, A_rows, dACsrRowPtr);

  alphasparseSpMatDescr_t matA;
  alphasparseCreateCsr(&matA,
                       A_rows,
                       A_cols,
                       nnz,
                       dACsrRowPtr,
                       dAcol,
                       dAval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_32F);

  int *dBCsrRowPtr = NULL;
  int *dBrow = NULL;
  int *dBcol = NULL;
  float *dBval = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBrow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dBval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(
      hipMalloc((void **)&dBCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
      dBrow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
      dBcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
      hipMemcpy(dBval, coo_values, nnz * sizeof(float), hipMemcpyHostToDevice));
  alphasparseXcoo2csr(dBrow, nnz, A_rows, dBCsrRowPtr);

  alphasparseSpMatDescr_t matB;
  alphasparseCreateCsr(&matB,
                       A_rows,
                       A_cols,
                       nnz,
                       dBCsrRowPtr,
                       dBcol,
                       dBval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_32F);

  int *dCCsrRowPtr = NULL;
  int *dCrow = NULL;
  int *dCcol = NULL;
  float *dCval = NULL;

  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dCrow, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dCcol, sizeof(int) * nnz));
  PRINT_IF_HIP_ERROR(hipMalloc((void **)&dCval, sizeof(float) * nnz));
  PRINT_IF_HIP_ERROR(
      hipMalloc((void **)&dCCsrRowPtr, sizeof(int) * (A_rows + 1)));
  CHECK_CUDA(hipMemcpy(
      dCrow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(hipMemcpy(
      dCcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK_CUDA(
      hipMemcpy(dCval, coo_values, nnz * sizeof(float), hipMemcpyHostToDevice));
  // alphasparseXcoo2csr(dCrow, nnz, A_rows, dCCsrRowPtr);

  alphasparseSpMatDescr_t matC;
  alphasparseCreateCsr(&matC,
                       A_rows,
                       A_cols,
                       nnz,
                       dCCsrRowPtr,
                       dCcol,
                       dCval,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEXTYPE_I32,
                       ALPHA_SPARSE_INDEX_BASE_ZERO,
                       ALPHA_R_32F);

  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;
  void *dBuffer1 = NULL;
  void *dBuffer2 = NULL;
  // SpGEMM Computation
  alphasparseSpGEMMDescr_t spgemmDesc;
  alphasparseSpGEMM_createDescr(&spgemmDesc);
  hipEvent_t event_start, event_stop;
  float elapsed_time = 0.0;
  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  alphasparseSpGEMM_compute(handle,
                            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            matA,
                            matB,
                            &beta,
                            matC,
                            ALPHA_R_32F,
                            ALPHASPARSE_SPGEMM_DEFAULT,
                            spgemmDesc,
                            &bufferSize2,
                            NULL);
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  std::cout << "alphasparseSpGEMM_compute1 time: " << elapsed_time << " microseconds, buffersize :" << bufferSize2 <<std::endl;
  double time1 = get_time_us();
  hipMalloc(&dBuffer2, bufferSize2);
  double time2 = get_time_us();
  std::cout << "alphasparseSpGEMM buffer malloc time: " << (time2 - time1) / (1e3) << " microseconds" << std::endl;
  double etime = elapsed_time + (time2 - time1) / (1e3);
  alphasparseSpGEMM_compute(handle,
                            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                            ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha,
                            matA,
                            matB,
                            &beta,
                            matC,
                            ALPHA_R_32F,
                            ALPHASPARSE_SPGEMM_DEFAULT,
                            spgemmDesc,
                            &bufferSize2,
                            dBuffer2);
  // allocate matrix C
  nnz_c = matC->nnz;
  matC_ict = (float*)alpha_malloc(matC->nnz * sizeof(float));
  CHECK_CUDA(hipMemcpy(matC_ict,
                        matC->val_data,
                        matC->nnz * sizeof(float),
                        hipMemcpyDeviceToHost))

  for (int i = 0; i < warm_up; i++)
  {
    alphasparseSpGEMM_compute(handle,
                              ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                              ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              matA,
                              matB,
                              &beta,
                              matC,
                              ALPHA_R_32F,
                              ALPHASPARSE_SPGEMM_DEFAULT,
                              spgemmDesc,
                              &bufferSize2,
                              dBuffer2);
  }

  GPU_TIMER_START(elapsed_time, event_start, event_stop);
  for (int i = 0; i < trials; i++)
  {
    alphasparseSpGEMM_compute(handle,
                              ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                              ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha,
                              matA,
                              matB,
                              &beta,
                              matC,
                              ALPHA_R_32F,
                              ALPHASPARSE_SPGEMM_DEFAULT,
                              spgemmDesc,
                              &bufferSize2,
                              dBuffer2);
  }
  GPU_TIMER_END(elapsed_time, event_start, event_stop);
  std::cout << "alphasparseSpGEMM_compute2 time: " << elapsed_time/trials<< " microseconds" << std::endl;
  
  // Clear up on device
  hipFree(dArow);
  hipFree(dAcol);
  hipFree(dAval);
  hipFree(dBrow);
  hipFree(dBcol);
  hipFree(dBval);
  hipFree(dCrow);
  hipFree(dCcol);
  hipFree(dCval);
  hipFree(dBuffer2);
  alphasparse_destory_handle(handle);
}

int main(int argc, const char *argv[])
{
  args_help_hip(argc, argv);
  file = args_get_data_file(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  transA = alpha_args_get_transA(argc, argv);
  transB = alpha_args_get_transB(argc, argv);
  trials = args_get_iter(argc, argv);
  // read coo
  alpha_read_coo<float>(
      file, &A_rows, &A_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
  coo_order<int32_t, float>(rnnz, coo_row_index, coo_col_index, coo_values);
  columns = args_get_cols(argc, argv, A_rows); // 默认C是方阵
  printf("rnnz %d\n", rnnz);
  for (int i = 0; i < 20; i++)
  {
    std::cout << coo_row_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++)
  {
    std::cout << coo_col_index[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++)
  {
    std::cout << coo_values[i] << ", ";
  }
  std::cout << std::endl;

  cuda_mm();
  alpha_mm();
  // cpu_mm();
  std::cout << std::endl;
  for (int i = 0; i < 20; i++)
  {
    std::cout << matC_cuda[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < 20; i++)
  {
    std::cout << matC_ict[i] << ", ";
  }
  check((float *)matC_cuda, nnz_c, (float *)matC_ict, nnz_c);
  // std::cout << std::endl << "check" << std::endl;
  // for (int i = 0; i < nnz_c; i++)
  // {
  //   if((matC_ict[i] - matC_roc[i])/matC_roc[i] > 1e-5)
  //     std::cout << i << ":" << matC_ict[i] << ", " << matC_roc[i] << std::endl;
  // }
  return 0;
}
