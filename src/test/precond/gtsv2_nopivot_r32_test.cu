#include "../test_common.h"

/**
 * @brief ict dcu mv hyb test
 * @author HPCRC, ICT
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "alphasparse.h"

bool check_flag;

alphasparseOperation_t transA;
alphasparseDirection_t dir_alpha;

int m = 1024, n = 1024, ldb, size;
float *hdl, *hd, *hdu, *hictB, *hcudaB;

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

void
cuda_gtsv2()
{
  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle));

  float* ddl = NULL;
  float* dd = NULL;
  float* ddu = NULL;
  float* dB = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(float) * m))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(float) * m))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(float) * m))
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(float) * size))

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(ddl, hdl, sizeof(float) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dd, hd, sizeof(float) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(ddu, hdu, sizeof(float) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dB, hcudaB, sizeof(float) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  CHECK_CUSPARSE(cusparseSgtsv2_nopivot_bufferSizeExt(
    handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size))

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));
  CHECK_CUSPARSE(
    cusparseSgtsv2_nopivot(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer));

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
    cudaMemcpy(hcudaB, dB, sizeof(float) * size, cudaMemcpyDeviceToHost));

  CHECK_CUSPARSE(cusparseDestroy(handle))
  CHECK_CUDA(cudaFree(ddl))
  CHECK_CUDA(cudaFree(dd))
  CHECK_CUDA(cudaFree(ddu))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(temp_buffer))
}

void
alpha_gtsv2()
{
  alphasparseHandle_t handle;
  initHandle(&handle);
  alphasparseGetHandle(&handle);

  float* ddl = NULL;
  float* dd = NULL;
  float* ddu = NULL;
  float* dB = NULL;

  CHECK_CUDA(cudaMalloc((void**)&ddl, sizeof(float) * m))
  CHECK_CUDA(cudaMalloc((void**)&dd, sizeof(float) * m))
  CHECK_CUDA(cudaMalloc((void**)&ddu, sizeof(float) * m))
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(float) * size))

  // Copy data to device
  CHECK_CUDA(cudaMemcpy(ddl, hdl, sizeof(float) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(dd, hd, sizeof(float) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(ddu, hdu, sizeof(float) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(
    cudaMemcpy(dB, hictB, sizeof(float) * size, cudaMemcpyHostToDevice))

  // Obtain required buffer size
  size_t buffer_size;
  alphasparseSgtsv2_nopivot_bufferSizeExt(
    handle, m, n, ddl, dd, ddu, dB, ldb, &buffer_size);

  void* temp_buffer;
  CHECK_CUDA(cudaMalloc(&temp_buffer, buffer_size));

  alphasparseSgtsv2_nopivot(handle, m, n, ddl, dd, ddu, dB, ldb, temp_buffer);

  // Device synchronization
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(
    cudaMemcpy(hictB, dB, sizeof(float) * size, cudaMemcpyDeviceToHost));

  alphasparse_destory_handle(handle);
  CHECK_CUDA(cudaFree(ddl))
  CHECK_CUDA(cudaFree(dd))
  CHECK_CUDA(cudaFree(ddu))
  CHECK_CUDA(cudaFree(dB))
  CHECK_CUDA(cudaFree(temp_buffer))
}

int
main(int argc, const char* argv[])
{
  // args
  args_help(argc, argv);
  check_flag = args_get_if_check(argc, argv);
  m = args_get_rows(argc, argv, m);
  n = args_get_cols(argc, argv, n);
  ldb = m;
  size = ldb * n;
  hdl = (float*)alpha_malloc(m * sizeof(float));
  hd = (float*)alpha_malloc(m * sizeof(float));
  hdu = (float*)alpha_malloc(m * sizeof(float));
  hcudaB = (float*)alpha_malloc(size * sizeof(float));
  hictB = (float*)alpha_malloc(size * sizeof(float));

  alpha_fill_random(hdl, 2, m);
  hdl[0] = 0.f;
  alpha_fill_random(hd, 3, m);
  alpha_fill_random(hdu, 4, m);
  hdu[m - 1] = 0.f;
  alpha_fill_random(hcudaB, 1, size);
  alpha_fill_random(hictB, 1, size);
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcudaB[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hictB[i] << ", ";
  }
  std::cout << std::endl;
  cuda_gtsv2();
  alpha_gtsv2();

  for (int i = 0; i < min(20, size); i++) {
    std::cout << hcudaB[i] << ", ";
  }
  std::cout << std::endl;
  for (int i = 0; i < min(20, size); i++) {
    std::cout << hictB[i] << ", ";
  }
  std::cout << std::endl;

  check(hictB, size, hcudaB, size);

  return 0;
}
