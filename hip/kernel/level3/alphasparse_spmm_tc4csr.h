#include "hip/hip_runtime.h"
#include "alphasparse.h"
#include <iostream>
#include <stdlib.h>

namespace {
struct bcsr_format {
  void* bcsrValues;
  void* bcsrRowPtr;
  void* bcsrColIdx;
  int32_t nonzeroBlocks;
  void* blockInfo;
  void* relativeBlockIndexMapping;
  int32_t MMA_M;
  int32_t MMA_N;
  int32_t MMA_K;
  bool valid = false;
};
bcsr_format bcsrMat;

} // namespace

template<typename T>
struct mma_helper {
  using VType = T;
};

template<>
struct mma_helper<float> {
  using VType = __attribute__((__vector_size__(4 * sizeof(float)))) float;
};

template<>
struct mma_helper<double> {
  using VType = __attribute__((__vector_size__(4 * sizeof(double)))) double;
};

template<typename U,
         typename T,
         unsigned int MMA_M,
         unsigned int MMA_N,
         unsigned int MMA_K>
void tc_preprocess(
  T* __restrict__ csr_row_ptr, // device
  T* __restrict__ csr_col_ind, // device
  U* __restrict__ csr_val, // device
  T M, T K, T nnz)
{
  // copy to host
  T *csr_row_ptr_h = (T*)malloc(sizeof(T) * (M + 1));
  T *csr_col_ind_h = (T*)malloc(sizeof(T) * nnz);
  U *csr_val_h = (U*)malloc(sizeof(U) * nnz);
  hipMemcpy(csr_row_ptr_h, csr_row_ptr, sizeof(T) * (M + 1), hipMemcpyDeviceToHost);
  hipMemcpy(csr_col_ind_h, csr_col_ind, sizeof(T) * nnz, hipMemcpyDeviceToHost);
  hipMemcpy(csr_val_h, csr_val, sizeof(U) * nnz, hipMemcpyDeviceToHost);

  T numColRegions = (K + MMA_K - 1) / MMA_K;
  T numRowRegions = (M + MMA_M - 1) / MMA_M;

  T numberOfBlocks = numRowRegions * numColRegions;
  T nonzeroBlocks = 0;
  
  T *blockInfo_host = (T*) calloc(sizeof(T), numberOfBlocks);
  // 0 - zero block
  // 1 - sparse block
  for (T row = 0; row < M; row++)
  {
    for (T j = csr_row_ptr[row]; j < csr_row_ptr[row + 1]; j++) 
    {
      T col = csr_col_ind_h[j];
      T rowRegion = row / MMA_M;
      T colRegion = col / MMA_K;
      T blockIndex = rowRegion * numColRegions + colRegion;
      if (blockInfo_host[blockIndex] == 0)  // zero block, stops being 0, becomes sparse
      {
        blockInfo_host[blockIndex] = 1;
        nonzeroBlocks += 1;
      }
    }
  }

  T relativeIndex = 0;
  T *relativeBlockIndexMapping_host = (T*) malloc(numberOfBlocks * sizeof(T));
  for (T i = 0; i < numberOfBlocks; i++)
  {
    relativeBlockIndexMapping_host[i] = (blockInfo_host[i] != 0) ? relativeIndex++ : -1;
    //printf("relative [%d] = %d\n", i, relativeBlockIndexMapping[i]);
  }

  // get the bcsr
  T *bcsrRowPtr_host = (T*)calloc(sizeof(T), (M / MMA_M + 1));
  T *bcsrColIdx_host = (T*)malloc(nonzeroBlocks * sizeof(T));
  U *bcsrVal_host = (U*)calloc(sizeof(U), nonzeroBlocks * MMA_M * MMA_K);

  T num_blocks = 0;
  
  // Do the rowPtrBcsr and colIdxBcsr
  for (T row = 0; row < M; row += MMA_M) {
    bcsrRowPtr_host[row / MMA_M] = num_blocks; // Update rowPtr

    for (T col = 0; col < K; col += MMA_K) {
      T current_block = row / MMA_M * numColRegions + col / MMA_K;
      if (blockInfo_host[current_block] == 0)
      {
          continue;
      }
      bcsrColIdx_host[num_blocks] = col; // not relative bcsr columns index / MMA_K if want relative
      num_blocks++;
    }
  }

  bcsrRowPtr_host[M / MMA_M] = num_blocks; // Update last entry of rowPtr
  
  // Do the valuesBcsr
  for (T row = 0; row < M; row++)
  {
      for (T j = csr_row_ptr_h[row]; j < csr_row_ptr_h[row + 1]; j++) 
      {
          T col = csr_col_ind_h[j];
          T rowRegion = row / MMA_M;
          T colRegion = col / MMA_K;
          T blockIndex = rowRegion * numColRegions + colRegion;
          U val = csr_val_h[j];
          T offset = row % MMA_M * MMA_K + col % MMA_K;
          T bcsrIndex = relativeBlockIndexMapping_host[blockIndex] * MMA_M * MMA_K + offset;
          bcsrVal_host[bcsrIndex] = val;
      }
  }
  T *bcsrRowPtr, *bcsrColIdx, *blockInfo, *relativeBlockIndexMapping;
  U *bcsrVal;
  hipMalloc((void**)&bcsrRowPtr, sizeof(T) * (M / MMA_M + 1));
  hipMalloc((void**)&bcsrColIdx, sizeof(T) * nonzeroBlocks);
  hipMalloc((void**)&bcsrVal, sizeof(U) * nonzeroBlocks * MMA_M * MMA_K);
  hipMalloc((void**)&blockInfo, sizeof(T) * numberOfBlocks);
  hipMalloc((void**)&relativeBlockIndexMapping, sizeof(T) * numberOfBlocks);

  hipMemcpy(bcsrRowPtr, bcsrRowPtr_host, sizeof(T) * (M / MMA_M + 1), hipMemcpyHostToDevice);
  hipMemcpy(bcsrColIdx, bcsrColIdx_host, sizeof(T) * nonzeroBlocks, hipMemcpyHostToDevice);
  hipMemcpy(bcsrVal, bcsrVal_host, sizeof(U) * nonzeroBlocks * MMA_M * MMA_K, hipMemcpyHostToDevice);
  hipMemcpy(blockInfo, blockInfo_host, sizeof(T) * numberOfBlocks, hipMemcpyHostToDevice);
  hipMemcpy(relativeBlockIndexMapping, relativeBlockIndexMapping_host, sizeof(T) * numberOfBlocks, hipMemcpyHostToDevice);

  bcsrMat.bcsrValues = bcsrVal;
  bcsrMat.bcsrRowPtr = bcsrRowPtr;
  bcsrMat.bcsrColIdx = bcsrColIdx;
  bcsrMat.blockInfo = blockInfo;
  bcsrMat.relativeBlockIndexMapping = relativeBlockIndexMapping;
  bcsrMat.nonzeroBlocks = nonzeroBlocks;
  bcsrMat.MMA_M = MMA_M;
  bcsrMat.MMA_N = MMA_N;
  bcsrMat.MMA_K = MMA_K;
  bcsrMat.valid = true;
}

template<unsigned int block_size,
         unsigned int warp_size,
         typename T,
         typename U,
         typename V,
         typename W,
         unsigned int MMA_M,
         unsigned int MMA_N,
         unsigned int MMA_K>
__global__ void tc_kernel(T M, T N, T K, T nnz, W alpha,
                          const T* __restrict__ bcsrRowPtr,
                          const T* __restrict__ bcsrColIdx,
                          const U* __restrict__ bcsrValues,
                          const T* __restrict__ blockInfo,
                          const T* __restrict__ relativeBlockIndexMapping,
                          T nonzeroBlocks,
                          const U* __restrict__ matB,
                          T ldb, W beta,
                          V* __restrict__ matC,
                          T ldc)
{
  const T K_tiles = CEIL(K, MMA_K);

  const T warp_row = blockIdx.y * MMA_M;
  const T warp_col = blockIdx.x * MMA_N;

  T blockRow = blockIdx.y;
  T blockCol = blockIdx.x;

  T colRegions = K_tiles;

  if (warp_row >= M || warp_col >= N) return;

  const T warp_laneid = threadIdx.x & (warp_size - 1);

  T local_row_a = warp_laneid % MMA_M;
  T local_col_a = warp_laneid / MMA_M;

  T local_row_b = warp_laneid / MMA_M;
  T local_col_b = warp_laneid % MMA_M;

  W frag_a = {};
  W frag_b = {};
  using VW = typename mma_helper<W>::VType;
  VW frag_c = {};
#pragma unroll
  for (T ptr = bcsrRowPtr[blockRow]; ptr < bcsrRowPtr[blockRow + 1]; ptr++) {
    T i = bcsrColIdx[ptr] / MMA_K;
    T blockIndex = blockRow * colRegions + i;
    T relativeIndex = relativeBlockIndexMapping[blockIndex];
    frag_a = bcsrValues[relativeIndex * MMA_M * MMA_K + local_row_a * MMA_K + local_col_a];
    frag_b = matB[(i * MMA_K + local_row_b) * ldb + warp_col + local_col_b];

    if constexpr (std::is_same<W, float>::value) {
      frag_c = __builtin_amdgcn_mmac_f32_16x16x4f32(frag_a, frag_b, frag_c, 0);
    }
    else if constexpr (std::is_same<W, double>::value) {
      frag_c = __builtin_amdgcn_mmac_f64_16x16x4f64(frag_a, frag_b, frag_c, 0);
    }
    else {
      ;
    }
  }
  #pragma unroll
  for (int k = 0; k < 4; k++) {
    matC[(warp_row + local_row_a) * ldc + warp_col + local_col_a + k * 4] = 
      alpha * frag_c[k] + beta * matC[(warp_row + local_row_a) * ldc + warp_col + local_col_a + k * 4];
  }
}

template<typename T, typename U, typename V, typename W,
         unsigned int MMA_M, unsigned int MMA_N, unsigned int MMA_K>
alphasparseStatus_t
csrspmm_tc(alphasparseHandle_t handle,
            T M, T N, T K, T nnz, W alpha, 
            const T* __restrict__ csr_row_ptr,
            const T* __restrict__ csr_col_ind,
            const U* __restrict__ csr_val,
            const U* __restrict__ matB,
            T ldb,  W beta,
            V* __restrict__ matC,
            T ldc,
            void* externalBuffer)
{
  constexpr int warp_size = 64;
  if (!bcsrMat.valid) {
    std::cout << "need to call alphasparseSPMM_preprocess for ALPHASPARSE_SPMM_CSR_ALG6!" << std::endl;
  }
  if (bcsrMat.MMA_M != MMA_M || bcsrMat.MMA_N != MMA_N || bcsrMat.MMA_K != MMA_K) {
    std::cout << "need to call preprocess for a new sparse matrix!" << std::endl;
    return ALPHA_SPARSE_STATUS_EXECUTION_FAILED;
  }

  T rowBlocks = CEIL(M, MMA_M);
  T colBlocks = CEIL(N, MMA_N);
  dim3 block(warp_size);
  dim3 grid(CEIL(N, MMA_N), CEIL(M, MMA_M));
  tc_kernel<64, 64, T, U, V, W, MMA_M, MMA_N, MMA_K><<<grid, block, 0, handle->stream>>>(M, N, K, nnz, alpha,
    (T*)bcsrMat.bcsrRowPtr, (T*)bcsrMat.bcsrColIdx, (U*)bcsrMat.bcsrValues, (T*)bcsrMat.blockInfo,
    (T*)bcsrMat.relativeBlockIndexMapping, (T)bcsrMat.nonzeroBlocks, matB, ldb, beta, matC, ldc);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}