#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#include <stdlib.h>

template <typename TYPE>
alphasparseStatus_t symv_x_csr_n_hi_omp(const TYPE alpha, const internal_spmat A,
                                               const TYPE *x, const TYPE beta,
                                               TYPE *y) {
  const ALPHA_INT m = A->rows;
  const ALPHA_INT n = A->cols;
  ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for (ALPHA_INT i = 0; i < m; ++i) {
    y[i] = alpha_mul(y[i], beta);
  }

  ALPHA_INT ldy_local = m;
  TYPE *y_local_total =
      (TYPE *)alpha_memalign(ldy_local * num_threads * sizeof(TYPE), DEFAULT_ALIGNMENT);
  memset(y_local_total, '\0', ldy_local * num_threads * sizeof(TYPE));

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    TYPE *y_local = &y_local_total[tid * ldy_local];

#ifdef _OPENMP
#pragma omp for
#endif
    for (ALPHA_INT r = 0; r < m; ++r) {
      TYPE tmp;
      tmp = alpha_setzero(tmp);
      ALPHA_INT rs = A->row_data[r];
      ALPHA_INT re = A->row_data[r+1];
      ALPHA_INT start = alpha_lower_bound(&A->col_data[rs], &A->col_data[re], r) - A->col_data;
      if (start < re && A->col_data[start] == r) {
        tmp = alpha_mul(alpha, ((TYPE *)A->val_data)[start]);
        y_local[r] = alpha_madd(tmp, x[r], y_local[r]);
        start += 1;
      }
      ALPHA_INT end = re;
      ALPHA_INT nnzr = end - start;
      for (int ai = start; ai < end; ai++) {
        tmp = alpha_madd(((TYPE *)A->val_data)[ai], x[A->col_data[ai]], tmp);
      }
      // tmp = vec_doti(nnzr, &((TYPE *)A->val_data)[start], &A->col_data[start], x);
      y_local[r] = alpha_madd(tmp, alpha, y_local[r]);

      ALPHA_INT *A_col = &A->col_data[start];
      TYPE *A_val = &((TYPE *)A->val_data)[start];
      ALPHA_INT nnzr4 = nnzr - 3;
      ALPHA_INT i = 0;
      tmp = alpha_mul(alpha, x[r]);
      // vec_fmai(y_local, A_col, A_val, nnzr, tmp);
      for (; i < nnzr4; i += 4) {
        y_local[A_col[i]] = alpha_madd(tmp, A_val[i], y_local[A_col[i]]);
        y_local[A_col[i + 1]] = alpha_madd(tmp, A_val[i + 1], y_local[A_col[i + 1]]);
        y_local[A_col[i + 2]] = alpha_madd(tmp, A_val[i + 2], y_local[A_col[i + 2]]);
        y_local[A_col[i + 3]] = alpha_madd(tmp, A_val[i + 3], y_local[A_col[i + 3]]);
      }
      for (; i < nnzr; i += 1) {
        y_local[A_col[i]] = alpha_madd(tmp, A_val[i], y_local[A_col[i]]);
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT low = cross_block_low(tid, num_threads, m);
    ALPHA_INT high = cross_block_high(tid, num_threads, m);
    for (ALPHA_INT i = 0; i < num_threads; i++) {
      TYPE *y_local = &y_local_total[i * ldy_local];
      ALPHA_INT row = low;
      for (; row < high - 3; row += 4) {
        y[row] = alpha_add(y[row], y_local[row]);
        y[row + 1] = alpha_add(y[row + 1], y_local[row + 1]);
        y[row + 2] = alpha_add(y[row + 2], y_local[row + 2]);
        y[row + 3] = alpha_add(y[row + 3], y_local[row + 3]);
      }
      for (; row < high; row += 1) {
        y[row] = alpha_add(y[row], y_local[row]);
      }
    }
  }

  alpha_free(y_local_total);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t symv_x_csr_n_hi_omp_serial(const TYPE alpha,
                                                      const internal_spmat A, const TYPE *x,
                                                      const TYPE beta, TYPE *y) {
  const ALPHA_INT m = A->rows;
  const ALPHA_INT n = A->cols;
  ALPHA_INT num_threads = alpha_get_thread_num();

  ALPHA_INT *start = (ALPHA_INT *)malloc(m * sizeof(ALPHA_INT));
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for (ALPHA_INT r = 0; r < m; ++r) {
    y[r] = alpha_mul(y[r], beta);
    TYPE tmp;
    ALPHA_INT rs = A->row_data[r];
    ALPHA_INT re = A->row_data[r+1];
    start[r] = alpha_lower_bound(&A->col_data[rs], &A->col_data[re], r) - A->col_data;
    if (start[r] < re && A->col_data[start[r]] == r) {
      tmp = alpha_mul(alpha, ((TYPE *)A->val_data)[start[r]]);
      y[r] = alpha_madd(tmp, x[r], y[r]);
      start[r] += 1;
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
  for (ALPHA_INT r = 0; r < m; ++r) {
    // TYPE tmp = vec_doti(end - start[r], &((TYPE *)A->val_data)[start[r]], &A->col_data[start[r]], x);
    TYPE tmp;
    tmp = alpha_setzero(tmp);
    for (int ai = start[r]; ai <  A->row_data[r+1]; ai++) {
      tmp = alpha_madd(((TYPE *)A->val_data)[ai], x[A->col_data[ai]], tmp);
    }
    y[r] = alpha_madd(tmp, alpha, y[r]);
  }
  for (ALPHA_INT r = 0; r < m; ++r) {
    ALPHA_INT end = A->row_data[r+1];
    ALPHA_INT *A_col = &A->col_data[start[r]];
    TYPE *A_val = &((TYPE *)A->val_data)[start[r]];
    ALPHA_INT nnzr = end - start[r];
    ALPHA_INT nnzr4 = nnzr - 3;
    ALPHA_INT i = 0;
    TYPE tmp;
    tmp = alpha_mul(alpha, x[r]);
    for (; i < nnzr4; i += 4) {
      y[A_col[i]] = alpha_madd(tmp, A_val[i], y[A_col[i]]);
      y[A_col[i + 1]] = alpha_madd(tmp, A_val[i + 1], y[A_col[i + 1]]);
      y[A_col[i + 2]] = alpha_madd(tmp, A_val[i + 2], y[A_col[i + 2]]);
      y[A_col[i + 3]] = alpha_madd(tmp, A_val[i + 3], y[A_col[i + 3]]);
    }
    for (; i < nnzr; i += 1) {
      y[A_col[i]] = alpha_madd(tmp, A_val[i], y[A_col[i]]);
    }
  }
  alpha_free(start);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t
symv_csr_n_hi(const TYPE alpha, const internal_spmat A, const TYPE *x,
                          const TYPE beta, TYPE *y) {
  if (A->row_data[A->rows] / (A->rows * A->cols) > 0.01) {
    return symv_x_csr_n_hi_omp(alpha, A, x, beta, y);
  } else {
    return symv_x_csr_n_hi_omp_serial(alpha, A, x, beta, y);
  }
}
