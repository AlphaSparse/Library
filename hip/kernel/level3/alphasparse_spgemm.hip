#include <new>
#include "alphasparse_spgemm_nnz_csr.h"
#include "alphasparse_spgemm_copy_csr.h"
#include "alphasparse_spgemm_csr.h"
#include "alphasparse_spgemm_speck_csr.h"

#include <iostream>
#include "../some_common.h"

static constexpr int spECK_STATIC_MEM_PER_BLOCK{49152};
static constexpr int spECK_DYNAMIC_MEM_PER_BLOCK{98304}; // V100

// template <typename T>
// __host__ __device__ __forceinline__ T divup(T a, T b)
// {
//   return (a + b - 1) / b;
// }

size_t get_dataSize(alphasparseDataType D)
{
  switch (D)
  {
  case ALPHA_R_32F:
    return sizeof(float);
  case ALPHA_R_64F:
    return sizeof(double);
  case ALPHA_C_32F:
    return sizeof(float) * 2;
  case ALPHA_C_64F:
    return sizeof(double) * 2;
  case ALPHA_R_16F:
    return sizeof(float);
  case ALPHA_C_16F:
    return sizeof(double);
  default:
    return sizeof(double);
  }
}

template <typename T, typename U>
alphasparseStatus_t
spgemm_template(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                alphasparseOperation_t opB,
                const void *alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseSpMatDescr_t matB,
                const void *beta,
                alphasparseSpMatDescr_t matC,
                void *externalBuffer2)
{
  switch (matA->format)
  {
  case ALPHA_SPARSE_FORMAT_CSR:
  {
    T nnz;
    spgemm_nnz_csr<T>(handle,
                      (T)matA->rows,
                      (T)matB->cols,
                      (T)matA->cols,
                      (T)matA->nnz,
                      (T *)matA->row_data,
                      (T *)matA->col_data,
                      (T)matB->nnz,
                      (T *)matB->row_data,
                      (T *)matB->col_data,
                      (T *)matC->row_data,
                      &nnz,
                      externalBuffer2);
    printf("nnzC %d \n", nnz);

    if (matC->nnz != nnz)
    {
      matC->nnz = nnz;
      hipMalloc((void **)&matC->col_data, sizeof(T) * nnz);
      hipMalloc((void **)&matC->val_data, sizeof(U) * nnz);
    }

    spgemm_csr<T,U>(handle,
                    (T)matA->rows,
                    (T)matB->cols,
                    (T)matA->cols,
                    *((U*)alpha),
                    (T*)matA->row_data,
                    (T*)matA->col_data,
                    (U*)matA->val_data,
                    (T)matA->nnz,
                    (T*)matB->row_data,
                    (T*)matB->col_data,
                    (U*)matB->val_data,
                    (T)matB->nnz,
                    *((U*)beta),   
                    (T*)matC->row_data,
                    (T*)matC->col_data,
                    (U*)matC->val_data,
                    externalBuffer2);

    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  }
  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

template <typename T, typename U>
alphasparseStatus_t
spgemm_speck_template(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      alphasparseOperation_t opB,
                      const void *alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseSpMatDescr_t matB,
                      const void *beta,
                      alphasparseSpMatDescr_t matC,
                      void *externalBuffer2)
{
  switch (matA->format)
  {
  case ALPHA_SPARSE_FORMAT_CSR:
  {
    spgemm_csr_spECK<T, U, 4, 1024, spECK_DYNAMIC_MEM_PER_BLOCK, spECK_STATIC_MEM_PER_BLOCK>(handle, opA, opB, *((U *)alpha), matA, matB, *((U *)beta), matC, externalBuffer2);
    break;
  }
  }
  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

template <typename T, typename U>
alphasparseStatus_t
spgemm_copy_template(alphasparseHandle_t handle,
                     alphasparseOperation_t opA,
                     alphasparseOperation_t opB,
                     const void *alpha,
                     alphasparseSpMatDescr_t matA,
                     alphasparseSpMatDescr_t matB,
                     const void *beta,
                     alphasparseSpMatDescr_t matC)
{
  switch (matA->format)
  {
  case ALPHA_SPARSE_FORMAT_CSR:
  {
    spgemm_copy_csr<T, U>(handle,
                          (T *)matC->row_data,
                          (T *)matC->col_data,
                          (U *)matC->val_data,
                          (T)matC->nnz,
                          *((U *)beta),
                          matC->idx_base);
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  }
  return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

alphasparseStatus_t
alphasparseSpGEMM_compute(alphasparseHandle_t handle,
                          alphasparseOperation_t opA,
                          alphasparseOperation_t opB,
                          const void *alpha,
                          alphasparseSpMatDescr_t matA,
                          alphasparseSpMatDescr_t matB,
                          const void *beta,
                          alphasparseSpMatDescr_t matC,
                          alphasparseDataType computeType,
                          alphasparseSpGEMMAlg_t alg,
                          alphasparseSpGEMMDescr_t spgemmDescr,
                          size_t *bufferSize2,
                          void *externalBuffer2)
{
  if (externalBuffer2 == NULL)
  {
    *bufferSize2 = 4;
    // for spECK
    size_t datasize = sizeof(double);
    size_t dataA = get_dataSize(matA->data_type);
    size_t dataB = get_dataSize(matB->data_type);
    size_t dataC = get_dataSize(matC->data_type);
    size_t dataCC = get_dataSize(computeType);
    datasize = max(dataA, max(dataB, max(dataC, dataCC)));
    // printf("datasize %d\n", datasize);
    size_t cubTempBytesScan = 0;
    void *cubTmp = nullptr;
    hipcub::DeviceScan::ExclusiveSum(cubTmp, cubTempBytesScan, (uint32_t *)matC->row_data, (uint32_t *)matC->row_data, matC->rows + 1);
    size_t buffer_size = cubTempBytesScan;
    size_t d_combined_pointers_size = sizeof(uint32_t) * (4 + 2 * matA->rows) + divup(cubTempBytesScan, sizeof(uint32_t)) * sizeof(uint32_t);
    if (matA->nnz > 10000)
      d_combined_pointers_size += sizeof(uint32_t) * matA->rows;
    buffer_size += d_combined_pointers_size;
    const int kernelCountNumeric = 6;
    const int kernelCountCounting = 6;
    uint32_t maxRowLength = max(1, (uint32_t)matB->cols * 12 / 10);
    const int staticSharedMemPerBlockCounting = 48, staticSharedMemPerBlockNumeric = 24;
    const int warpsNumeric = 1024 / 32;
    const int warpsCounting = 1024 / 32;
    const int sharedBytesPerWarpNumeric = spECK_STATIC_MEM_PER_BLOCK / warpsNumeric - staticSharedMemPerBlockNumeric;
    const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(uint32_t) + datasize);
    const int sharedBytesPerWarpCounting = spECK_STATIC_MEM_PER_BLOCK / warpsNumeric - staticSharedMemPerBlockCounting;
    const int entriesPerWarpCounting = sharedBytesPerWarpCounting / sizeof(uint32_t);
    int maxNnzPerBlockNumeric = entriesPerWarpNumeric * warpsNumeric * 2 / 3;
    int maxNnzPerBlockCounting = entriesPerWarpCounting * warpsCounting * 4 / 5;
    uint32_t actualKernelCount = min(kernelCountCounting,
                                     uint32_t(
                                         std::log2(
                                             divup(
                                                 int(maxRowLength),
                                                 min(
                                                     maxNnzPerBlockCounting >> (kernelCountCounting - 1),
                                                     maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)))) +
                                         1));
    size_t combinedBlockStartSize = sizeof(uint32_t) * (1 + kernelCountCounting + matA->rows * (1 + actualKernelCount));
    buffer_size += combinedBlockStartSize;
    buffer_size += sizeof(uint32_t);
    *bufferSize2 += buffer_size;
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F)
  {
    return spgemm_speck_template<uint32_t, float>(
    // return spgemm_template<int32_t, float>(
        handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F)
  {
    return spgemm_template<int32_t, double>(
        handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F) {
  //   return spgemm_opsp_template<uint32_t, hipFloatComplex>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F) {
  //   return spgemm_template<int32_t, hipDoubleComplex>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
  //     computeType == ALPHA_R_16F) {
  //   return spgemm_template<int32_t, half>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  // }
  // if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
  //     matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F) {
  //   return spgemm_template<int32_t, half2>(
  //     handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
  // }
// #if (CUDA_ARCH >= 80)
//   if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
//       matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF)
//   {
//     return spgemm_template<int32_t, nv_bfloat16>(
//         handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
//   }
//   if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
//       matA->data_type == ALPHA_C_16BF && matC->data_type == ALPHA_C_16BF)
//   {
//     return spgemm_template<int32_t, nv_bfloat162>(
//         handle, opA, opB, alpha, matA, matB, beta, matC, externalBuffer2);
//   }
// #endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpGEMM_copy(alphasparseHandle_t handle,
                       alphasparseOperation_t opA,
                       alphasparseOperation_t opB,
                       const void *alpha,
                       alphasparseSpMatDescr_t matA,
                       alphasparseSpMatDescr_t matB,
                       const void *beta,
                       alphasparseSpMatDescr_t matC,
                       alphasparseDataType computeType,
                       alphasparseSpGEMMAlg_t alg,
                       alphasparseSpGEMMDescr_t spgemmDescr)
{
  // single real ; i32
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_32F && matC->data_type == ALPHA_R_32F)
  {
    return spgemm_copy_template<int32_t, float>(
        handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_64F && matC->data_type == ALPHA_R_64F)
  {
    return spgemm_copy_template<int32_t, double>(
        handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_32F && matC->data_type == ALPHA_C_32F)
  {
    return spgemm_copy_template<int32_t, hipFloatComplex>(
        handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_64F && matC->data_type == ALPHA_C_64F)
  {
    return spgemm_copy_template<int32_t, hipDoubleComplex>(
        handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_R_16F && matC->data_type == ALPHA_R_16F &&
      computeType == ALPHA_R_16F)
  {
    return spgemm_copy_template<int32_t, half>(
        handle, opA, opB, alpha, matA, matB, beta, matC);
  }
  if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
      matA->data_type == ALPHA_C_16F && matC->data_type == ALPHA_C_16F)
  {
    return spgemm_copy_template<int32_t, half2>(
        handle, opA, opB, alpha, matA, matB, beta, matC);
  }
// #if (CUDA_ARCH >= 80)
//   if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
//       matA->data_type == ALPHA_R_16BF && matC->data_type == ALPHA_R_16BF)
//   {
//     return spgemm_copy_template<int32_t, nv_bfloat16>(
//         handle, opA, opB, alpha, matA, matB, beta, matC);
//   }
//   if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 &&
//       matA->data_type == ALPHA_C_16BF && matC->data_type == ALPHA_C_16BF)
//   {
//     return spgemm_copy_template<int32_t, nv_bfloat162>(
//         handle, opA, opB, alpha, matA, matB, beta, matC);
//   }
// #endif
  return ALPHA_SPARSE_STATUS_SUCCESS;
}