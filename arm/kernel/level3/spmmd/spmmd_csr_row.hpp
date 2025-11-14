#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/partition.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>

template <typename J>
static void spmmd_csr_row_unroll(const internal_spmat matA, const internal_spmat matB,
                                   J *matC, const ALPHA_INT ldc, ALPHA_INT lrs, ALPHA_INT lre) {
  for (ALPHA_INT ar = lrs; ar < lre; ar++) {
    J *tmpC = &matC[index2(ar, 0, ldc)];
    for (ALPHA_INT ai = matA->row_data[ar]; ai < matA->row_data[ar+1]; ai++) {
      ALPHA_INT br = matA->col_data[ai];
      J av = ((J*)matA->val_data)[ai];
      ALPHA_INT bis = matB->row_data[br];
      ALPHA_INT bie = matB->row_data[br+1];
      ALPHA_INT bil = bie - bis;
      const ALPHA_INT *B_col = &matB->col_data[bis];
      const J *B_val = &((J*)matB->val_data)[bis];
      ALPHA_INT bi = 0;
      for (; bi < bil - 3; bi += 4) {
        ALPHA_INT bc0 = B_col[bi];
        ALPHA_INT bc1 = B_col[bi + 1];
        ALPHA_INT bc2 = B_col[bi + 2];
        ALPHA_INT bc3 = B_col[bi + 3];
        J bv0 = B_val[bi];
        J bv1 = B_val[bi + 1];
        J bv2 = B_val[bi + 2];
        J bv3 = B_val[bi + 3];
        tmpC[bc0] = alpha_madd(av, bv0, tmpC[bc0]);
        tmpC[bc1] = alpha_madd(av, bv1, tmpC[bc1]);
        tmpC[bc2] = alpha_madd(av, bv2, tmpC[bc2]);
        tmpC[bc3] = alpha_madd(av, bv3, tmpC[bc3]);
      }
      for (; bi < bil; bi++) {
        ALPHA_INT bc = B_col[bi];
        J bv = B_val[bi];
        tmpC[bc] = alpha_madd(av, bv, tmpC[bc]);
      }
    }
  }
}

template <typename J>
alphasparseStatus_t spmmd_csr_row(const internal_spmat matA, const internal_spmat matB, J *matC,
                          const ALPHA_INT ldc) {
  ALPHA_INT m = matA->rows;
  ALPHA_INT num_thread = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
  for (ALPHA_INT i = 0; i < matA->rows; i++)
    for (ALPHA_INT j = 0; j < matB->cols; j++) {
      matC[index2(i, j, ldc)] = alpha_setzero(matC[index2(i, j, ldc)]);
    }

  ALPHA_INT64 flop[m];
  memset(flop, '\0', m * sizeof(ALPHA_INT64));

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
  for (ALPHA_INT ar = 0; ar < m; ar++) {
    for (ALPHA_INT ai = matA->row_data[ar]; ai < matA->row_data[ar+1]; ai++) {
      ALPHA_INT br = matA->col_data[ai];
      flop[ar] += matB->row_data[br+1] - matB->row_data[br];
    }
  }
  for (ALPHA_INT i = 1; i < m; i++) {
    flop[i] += flop[i - 1];
  }

  ALPHA_INT partition[num_thread + 1];
  balanced_partition_row_by_flop(flop, m, num_thread, partition);

  // 计算
#ifdef _OPENMP
#pragma omp parallel num_threads(num_thread)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT local_m_s = partition[tid];
    ALPHA_INT local_m_e = partition[tid + 1];
    spmmd_csr_row_unroll(matA, matB, matC, ldc, local_m_s, local_m_e);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
