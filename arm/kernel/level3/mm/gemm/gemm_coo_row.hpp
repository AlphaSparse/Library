#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename TYPE>
void mm_coo_outcols(const TYPE alpha, const internal_spmat mat,
                                 const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx,
                                 const TYPE beta, TYPE *y, const ALPHA_INT ldy,
                                 ALPHA_INT lrs, ALPHA_INT lre) {
  ALPHA_INT num_threads = alpha_get_thread_num();
  ALPHA_INT tid = alpha_get_thread_id();
  for (ALPHA_INT nn = lrs; nn < lre; ++nn) {
    ALPHA_INT cr = mat->row_data[nn];
    if (cr % num_threads != tid) continue;  // 相同行由相同线程处理

    TYPE *Y = &y[index2(cr, 0, ldy)];

    TYPE val;
    val = alpha_mul(alpha, ((TYPE *)mat->val_data)[nn]);
    const TYPE *X = &x[index2(mat->col_data[nn], 0, ldx)];
    ALPHA_INT c = 0;
    for (; c < columns; c++) {
      Y[c] = alpha_madde(Y[c], val, X[c]);
    }
  }
}

template <typename TYPE>
alphasparseStatus_t mm_coo_omp(const TYPE alpha, const internal_spmat mat,
                                      const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx,
                                      const TYPE beta, TYPE *y, const ALPHA_INT ldy) {
  ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for (int i = 0; i < mat->rows; i++)
    for (int j = 0; j < columns; j++) y[index2(i, j, ldy)] = alpha_mul(y[index2(i, j, ldy)], beta);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  { mm_coo_outcols(alpha, mat, x, columns, ldx, beta, y, ldy, 0, mat->nnz); }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemm_coo_row(const TYPE alpha, const internal_spmat mat, const TYPE *x,
                          const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta,
                          TYPE *y, const ALPHA_INT ldy) {
  return mm_coo_omp(alpha, mat, x, columns, ldx, beta, y, ldy);
}
