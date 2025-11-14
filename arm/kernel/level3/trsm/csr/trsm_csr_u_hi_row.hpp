#include <memory.h>

#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"
#include "alphasparse/util/pack.h"

#define COL_BLOCK 512

template <typename J>
alphasparseStatus_t trsm_csr_u_hi_row_opt(const J alpha, const internal_spmat A, const J *x,
                          const ALPHA_INT columns, const ALPHA_INT ldx, J *y,
                          const ALPHA_INT ldy) {
  ALPHA_INT m = A->rows;
  int num_thread = alpha_get_thread_num();
  const ALPHA_INT ldtmp = NPERCL;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_thread)
#endif
  for (ALPHA_INT cc = 0; cc < columns - NPERCL + 1; cc += NPERCL) {
    J *tmp_X =
        (J *)alpha_memalign(sizeof(J) * m * NPERCL, DEFAULT_ALIGNMENT);
    ALPHA_INT *row_start = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * m, DEFAULT_ALIGNMENT);
    ALPHA_INT *row_end = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * m, DEFAULT_ALIGNMENT);

    // m x 16行优先存储的X
    pack_r2r(m, NPERCL, x + cc, ldx, tmp_X, ldtmp);
    for (ALPHA_INT ac = m; ac > 0; ac -= COL_BLOCK) {
      const ALPHA_INT acs = alpha_max(0, ac - COL_BLOCK);
      const ALPHA_INT ace = ac;
      J *X = tmp_X + acs * ldtmp;
      const ALPHA_INT Xlen = (ace - acs) * ldtmp;
      // load对角块到cache
      // VEC_MUL2(X, X, alpha, Xlen);
      for (int i = 0; i < Xlen; i++) {
        X[i] = alpha_mul(X[i], alpha);
      }
      //取稀疏矩阵上三角的切片
      csr_uppercol_truncate(A, acs, ace, row_start, row_end);
      ALPHA_INT ar = ace - 1;
      //先求对角块COL_BLOCK x COL_BLOCK内的未知数
      for (; ar >= acs; ar--) {
        ALPHA_INT start = row_start[ar];
        ALPHA_INT end = row_end[ar];
        if (start < end && A->col_data[start] == ar) {
          start += 1;
        }
        J *dst = tmp_X + ar * ldtmp;
        for (ALPHA_INT nnz_indx = start; nnz_indx < end; nnz_indx++) {
          J val = ((J*)A->val_data)[nnz_indx];
          J *src = tmp_X + A->col_data[nnz_indx] * ldtmp;
          for (int c = 0; c < NPERCL; c++) {
            dst[c] = alpha_msub(src[c], val, dst[c]);
          }
          // VEC_FMS2(dst, src, val, NPERCL);
        }
      }
      // tmp_X[0:acs, 0:xcl] -= A[0:acs,acs:ace] * tmp_X[acs:ace, 0:xcl]
      for (ar = 0; ar < acs; ar++) {
        ALPHA_INT start = row_start[ar];
        ALPHA_INT end = row_end[ar];
        J *dst = tmp_X + ar * ldtmp;
        for (ALPHA_INT nnz_indx = start; nnz_indx < end; nnz_indx++) {
          J val = ((J*)A->val_data)[nnz_indx];
          J *src = tmp_X + A->col_data[nnz_indx] * ldtmp;
          for (int c = 0; c < NPERCL; c++) {
            dst[c] = alpha_msub(src[c], val, dst[c]);
          }
          // VEC_FMS2(dst, src, val, NPERCL);
        }
      }
    }
    pack_r2r(m, NPERCL, tmp_X, ldtmp, y + cc, ldy);
    alpha_free(tmp_X);
    alpha_free(row_end);
    alpha_free(row_start);
  }
  const ALPHA_INT cc = columns / NPERCL * NPERCL;
  const ALPHA_INT xcl = columns - cc;
  if (xcl > 0) {
    J *tmp_X =
        (J *)alpha_memalign(sizeof(J) * m * ldtmp, DEFAULT_ALIGNMENT);
    ALPHA_INT *row_start = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * m, DEFAULT_ALIGNMENT);
    ALPHA_INT *row_end = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * m, DEFAULT_ALIGNMENT);

    // m x 16行优先存储的X
    pack_r2r(m, xcl, x + cc, ldx, tmp_X, ldtmp);

    for (ALPHA_INT ac = m; ac > 0; ac -= COL_BLOCK) {
      const ALPHA_INT acs = alpha_max(0, ac - COL_BLOCK);
      const ALPHA_INT ace = ac;
      J *X = tmp_X + acs * ldtmp;
      const ALPHA_INT Xlen = (ace - acs) * ldtmp;
      // load对角块到cache
      // VEC_MUL2(X, X, alpha, Xlen);
      for (int c = 0; c < Xlen; c++) {
        X[c] = alpha_mul(X[c], alpha);
      }
      //取稀疏矩阵上三角的切片
      csr_uppercol_truncate(A, acs, ace, row_start, row_end);
      ALPHA_INT ar = ace - 1;
      //先求对角块COL_BLOCK x COL_BLOCK内的未知数
      for (; ar >= acs; ar--) {
        ALPHA_INT start = row_start[ar];
        ALPHA_INT end = row_end[ar];
        if (start < end && A->col_data[start] == ar) {
          start += 1;
        }
        J *dst = tmp_X + ar * ldtmp;
        for (ALPHA_INT nnz_indx = start; nnz_indx < end; nnz_indx++) {
          J val = ((J*)A->val_data)[nnz_indx];
          J *src = tmp_X + A->col_data[nnz_indx] * ldtmp;
          for (int c = 0; c < xcl; c++) {
            dst[c] = alpha_msub(src[c], val, dst[c]);
          }
          // VEC_FMS2(dst, src, val, xcl);
        }
      }
      // tmp_X[0:acs, 0:xcl] -= A[0:acs,acs:ace] * tmp_X[acs:ace, 0:xcl]
      for (ar = 0; ar < acs; ar++) {
        ALPHA_INT start = row_start[ar];
        ALPHA_INT end = row_end[ar];
        J *dst = tmp_X + ar * ldtmp;
        for (ALPHA_INT nnz_indx = start; nnz_indx < end; nnz_indx++) {
          J val = ((J*)A->val_data)[nnz_indx];
          J *src = tmp_X + A->col_data[nnz_indx] * ldtmp;
          for (int c = 0; c < xcl; c++) {
            dst[c] = alpha_msub(src[c], val, dst[c]);
          }
          // VEC_FMS2(dst, src, val, xcl);
        }
      }
    }
    pack_r2r(m, xcl, tmp_X, ldtmp, y + cc, ldy);
    alpha_free(tmp_X);
    alpha_free(row_end);
    alpha_free(row_start);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename J>
alphasparseStatus_t trsm_csr_u_hi_row(const J alpha, const internal_spmat A, const J *x,
                          const ALPHA_INT columns, const ALPHA_INT ldx, J *y,
                          const ALPHA_INT ldy) {
  ALPHA_INT m = A->rows;

  for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
  {
      for (ALPHA_INT r = m - 1; r >= 0; r--)
      {
          J temp;
          temp = alpha_setzero(temp);
          for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
          {
              ALPHA_INT ac = A->col_data[ai];
              if (ac > r)
              {
                temp = alpha_madde(temp, ((J*)A->val_data)[ai], y[ac * ldy + out_y_col]);
              }
          }
          J t;
          t = alpha_setzero(t);
          t = alpha_mul(alpha, x[r * ldx + out_y_col]);
          y[r * ldy + out_y_col] = alpha_sub(t, temp);
      }
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}