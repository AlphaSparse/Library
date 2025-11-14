#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/bisearch.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>
#ifdef __x86_64__
#include <immintrin.h>
#endif

#include <stdio.h>
#define CACHELINE 64
template <typename J>
alphasparseStatus_t
symm_csr_n_hi_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
  ALPHA_INT ncpu, mcpu, nblk, mblk, kblk;
  ALPHA_INT num_threads = alpha_get_thread_num();
  //2D partition
  balanced_divisors2(mat->rows, columns, num_threads, &mcpu, &ncpu);

  mblk = (mat->rows + mcpu - 1) / mcpu;
  nblk = (columns + ncpu - 1) / ncpu;
  // k blocking, for symm, kblk must be equal to mblk to check whether block of mat belongs to upper
  kblk = mblk;
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT tid_x = tid % mcpu;
    ALPHA_INT tid_y = tid / mcpu;
    //当前处理的矩阵块(tid_x, tid_y), 对应结果矩阵块[lrs:lre, lcs:lce]
    ALPHA_INT lrs = tid_x * mblk;
    ALPHA_INT lre = alpha_min((tid_x + 1) * mblk, mat->rows);
    ALPHA_INT lcs = tid_y * nblk;
    ALPHA_INT lce = alpha_min((tid_y + 1) * nblk, columns);
    const ALPHA_INT lcl = lce - lcs;

    for (ALPHA_INT ix = lrs; ix < lre; ++ix)
    {
      J *Y_dst = &y[index2(ix, lcs, ldy)];
      for (ALPHA_INT c = 0; c < lcl; c++)
      {
        Y_dst[c] = alpha_mul(Y_dst[c], beta);
      }
    }

    for (ALPHA_INT ki = 0; ki < mcpu; ki++)
    {
      ALPHA_INT ks = ki * kblk;
      ALPHA_INT ke = alpha_min((ki + 1) * kblk, mat->cols);
      if (ki > tid_x)
      { // upper part
        for (ALPHA_INT ar = lrs; ar < lre; ar++)
        {
          ALPHA_INT rs = mat->row_data[ar];
          ALPHA_INT re = mat->row_data[ar+1];
          ALPHA_INT start = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], ks) - mat->col_data;
          ALPHA_INT end = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], ke) - mat->col_data;

          for (ALPHA_INT ai = start; ai < end; ai++)
          {
            ALPHA_INT ac = mat->col_data[ai];
            J val = alpha_mul(((J*)mat->val_data)[ai], alpha);
            const J *X_src = &x[index2(ac, lcs, ldx)];
            J *Y_dst = &y[index2(ar, lcs, ldx)];

            for (ALPHA_INT c = 0; c < lcl; c++)
            {
              Y_dst[c] = alpha_madd(X_src[c], val, Y_dst[c]);
            }
          }
        }
      }
      else if (ki == tid_x)
      { // diagonal block
        for (ALPHA_INT ar = lrs; ar < lre; ar++)
        {
          ALPHA_INT rs = mat->row_data[ar];
          ALPHA_INT re = mat->row_data[ar+1];
          ALPHA_INT start = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], ks) - mat->col_data;
          ALPHA_INT end = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], ke) - mat->col_data;

          for (ALPHA_INT ai = start; ai < end; ai++)
          {
            ALPHA_INT ac = mat->col_data[ai];
            J val = alpha_mul(((J*)mat->val_data)[ai], alpha);
            if (ar < ac) // upper of diagnol block
            {
              const J *X_src = &x[index2(ac, lcs, ldx)];
              const J *Xsym_src = &x[index2(ar, lcs, ldx)];
              J *Y_dst = &y[index2(ar, lcs, ldx)];
              J *Ysym_dst = &y[index2(ac, lcs, ldx)];

              for (ALPHA_INT c = 0; c < lcl; c++)
              {
                Y_dst[c] = alpha_madd(X_src[c], val, Y_dst[c]);
              }
              for (ALPHA_INT c = 0; c < lcl; c++)
              {
                Ysym_dst[c] = alpha_madd(Xsym_src[c], val, Ysym_dst[c]);
              }
            }
            else if (ar == ac)
            {
              const J *X_src = &x[index2(ac, lcs, ldx)];
              J *Y_dst = &y[index2(ar, lcs, ldx)];

              for (ALPHA_INT c = 0; c < lcl; c++)
              {
                Y_dst[c] = alpha_madd(X_src[c], val, Y_dst[c]);
              }
            }
          }
        }
      }
      else
      { //lower part where nnz are found in the upper part
        for (ALPHA_INT ac = ks; ac < ke; ac++)
        {
          ALPHA_INT rs = mat->row_data[ac];
          ALPHA_INT re = mat->row_data[ac+1];
          ALPHA_INT start = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], lrs) - mat->col_data;
          ALPHA_INT end = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], lre) - mat->col_data;

          for (ALPHA_INT ai = start; ai < end; ai++)
          {
            ALPHA_INT ar = mat->col_data[ai];
            J val = alpha_mul(((J*)mat->val_data)[ai], alpha);
            const J *X_src = &x[index2(ac, lcs, ldx)];
            J *Y_dst = &y[index2(ar, lcs, ldx)];

            for (ALPHA_INT c = 0; c < lcl; c++)
            {
              Y_dst[c] = alpha_madd(X_src[c], val, Y_dst[c]);
            }
          }
        }
      }
    }
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
