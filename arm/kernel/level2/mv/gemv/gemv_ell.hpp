#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#include <string.h>

alphasparseStatus_t gemv_ell_row_partiton(
    const float alpha, const internal_spmat A, const ALPHA_INT lrs,
    const ALPHA_INT lre, const float *x, const float beta,
    float *y) {
    const ALPHA_INT m = A->rows;
    // const ALPHA_INT n = A->cols;
    const ALPHA_INT width = A->ell_width;
    const ALPHA_INT lrl = lre - lrs;
    float *y_local = (float *)alpha_malloc(sizeof(float) * lrl);
    memset(y_local, 0, sizeof(float) * lrl);
    for (ALPHA_INT ii = 0; ii < width; ii++)
    {
        float *values = &((float *)A->val_data)[ii * m];
        ALPHA_INT *indices = &A->ind_data[ii * m];

        __asm__ volatile(
            "prfm pldl3strm, [%[value_p]]\n\t"
            "prfm pldl3strm, [%[indx]]\n\t"
            "prfm pldl3strm, [%[y_p]]\n\t"
            :
            : [value_p] "r"(values + lrs), [indx] "r"(indices + lrs),
              [y_p] "r"(y_local));

        ALPHA_INT jj = lrs, jjl = 0;
        for (; jj < lre - 3; jj += 4, jjl += 4)
        {
            ALPHA_INT col0 = indices[jj + 0];
            ALPHA_INT col1 = indices[jj + 1];
            ALPHA_INT col2 = indices[jj + 2];
            ALPHA_INT col3 = indices[jj + 3];
  
            float val0 = values[jj + 0];
            float val1 = values[jj + 1];
            float val2 = values[jj + 2];
            float val3 = values[jj + 3];

            __asm__ volatile(
                "prfm pldl3strm, [%[value_p]]\n\t"
                "prfm pldl3strm, [%[indx]]\n\t"
                :
                : [value_p] "r"(values + jj + 4), [indx] "r"(indices + jj + 4));

            y_local[jjl + 0] = alpha_madde(y_local[jjl + 0], val0, x[col0]);
            y_local[jjl + 1] = alpha_madde(y_local[jjl + 1], val1, x[col1]);
            y_local[jjl + 2] = alpha_madde(y_local[jjl + 2], val2, x[col2]);
            y_local[jjl + 3] = alpha_madde(y_local[jjl + 3], val3, x[col3]);
        }
        for (; jj < lre; jj++, jjl++)
        {
            ALPHA_INT col0 = indices[jj + 0];
            float val0 = values[jj + 0];
            y_local[jjl + 0] = alpha_madde(y_local[jjl + 0], val0, x[col0]);
        }
    }
    for (ALPHA_INT jj = lrs, jjl = 0; jj < lre; jj++, jjl++)
    {
        y[jj] = alpha_mul(y[jj], beta);
        y[jj] = alpha_madde(y[jj], y_local[jjl], alpha);
    }
    alpha_free(y_local);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_ell_row_partiton(
    const TYPE alpha, const internal_spmat A, const ALPHA_INT lrs,
    const ALPHA_INT lre, const TYPE *x, const TYPE beta,
    TYPE *y) {
  const ALPHA_INT m = A->rows;
  const ALPHA_INT width = A->ell_width;
  const ALPHA_INT lrl = lre - lrs;

  for (ALPHA_INT r = lrs; r < lre; r++) {
    TYPE tmp;
    tmp = alpha_setzero(tmp);
    for (ALPHA_INT c = 0; c < width; c++) {
      TYPE val = ((TYPE *)A->val_data)[c * m + r];
      ALPHA_INT idx = A->ind_data[c * m + r];
      tmp = alpha_madd(val, x[idx], tmp);
    }
    y[r] = alpha_mul(y[r], beta);
    y[r] = alpha_madd(tmp, alpha, y[r]);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_ell_omp(const TYPE alpha,
                                         const internal_spmat mat,
                                         const TYPE *x,
                                         const TYPE beta,
                                         TYPE *y) {
  const ALPHA_INT rows = mat->rows;
  const ALPHA_INT cols = mat->cols;
  const ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    const ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT r_squeeze = (rows + NPERCL - 1) / NPERCL;
    ALPHA_INT lrs = (ALPHA_INT64)tid * r_squeeze / thread_num;
    ALPHA_INT lre = (ALPHA_INT64)(tid + 1) * r_squeeze / thread_num;
    lrs *= NPERCL;
    lre *= NPERCL;
    lrs = alpha_min(lrs, rows);
    lre = alpha_min(lre, rows);
    gemv_ell_row_partiton(alpha, mat, lrs, lre, x, beta, y);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
template <typename TYPE>
alphasparseStatus_t gemv_ell(const TYPE alpha, const internal_spmat mat,
                           const TYPE *x, const TYPE beta,
                           TYPE *y) {
  return gemv_ell_omp(alpha, mat, x, beta, y);
}
