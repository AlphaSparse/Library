#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_csr_vector.h"
#include "./line_enhance/line_enhance_spmv_imp.inl"

#define __WF_SIZE__ 32

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_line(alphasparseHandle_t handle,
                                  T m,
                                  T n,
                                  T nnz,
                                  const W alpha,
                                  const U *csr_val,
                                  const T *csr_row_ptr,
                                  const T *csr_col_ind,
                                  const U *x,
                                  const W beta,
                                  V *y,
                                  void *externalBuffer)
{
  constexpr int R = 2;
  constexpr int ROWS_PER_BLOCK = 32; // note: make sure ROWS_PER_BLOCK * VEC_SIZE <= THREADS_PER_BLOCK.

  constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
  constexpr int VEC_SIZE = 4; // note: if using direct reduce, VEC_SIZE must set to 1.

  int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
  constexpr int THREADS_PER_BLOCK = 512;
  LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_line_adaptive(alphasparseHandle_t handle,
                                           T m,
                                           T n,
                                           T nnz,
                                           const W alpha,
                                           const U *csr_val,
                                           const T *csr_row_ptr,
                                           const T *csr_col_ind,
                                           const U *x,
                                           const W beta,
                                           V *y,
                                           void *externalBuffer)
{
  // common parameters:
  constexpr int THREADS_PER_BLOCK = 512;

  // for small matrix.
  const int mtx_nnz = nnz;
  const int nnz_per_row = mtx_nnz / m;
  if (mtx_nnz <= (1 << 24))
  { // 2^24=16,777,216
    if (nnz_per_row >= 32)
    { // matrix has long rows
      constexpr int R = 4;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 8;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    }
    else
    { // matrix has show rows, then, use less thread(e.g. direct reduction) for reduction
      constexpr int R = 2;
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
  else
  { // for large matrix
    constexpr int R = 2;
    if (nnz_per_row >= 24)
    { // long row matrix
      constexpr int ROWS_PER_BLOCK = 64;
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_VEC;
      constexpr int VEC_SIZE = 4;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    }
    else
    {                                     // short row matrix, use direct reduce
      constexpr int ROWS_PER_BLOCK = 128; // more rows it can process
      constexpr int REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;
      constexpr int VEC_SIZE = 1;

      int BLOCKS = m / ROWS_PER_BLOCK + (m % ROWS_PER_BLOCK == 0 ? 0 : 1);
      LINE_ENHANCE_KERNEL_WRAPPER(REDUCE_OPTION, ROWS_PER_BLOCK, VEC_SIZE, R, BLOCKS, THREADS_PER_BLOCK);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
  }
}
