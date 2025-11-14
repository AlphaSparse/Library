#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include "alphasparse/util/partition.h"
#include "alphasparse/compute.h"
#include <type_traits>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <arm_neon.h>

#define VEC_MUL2_S(y, x, __val, len)                                    \
{                                                                       \
    float32x4_t y_v_0, x_v_0, y_v_1, x_v_1, y_v_2, x_v_2, y_v_3, x_v_3; \
    float32x4_t val_v;                                                  \
    val_v = vdupq_n_f32((__val));                                       \
    int32_t len16 = (len)-15;                                           \
    int32_t __i = 0;                                                    \
    for (; __i < len16; __i += 16)                                      \
    {                                                                   \
        x_v_0 = vld1q_f32((float *)(x + __i));                          \
        x_v_1 = vld1q_f32((float *)(x + __i + 4));                      \
        x_v_2 = vld1q_f32((float *)(x + __i + 8));                      \
        x_v_3 = vld1q_f32((float *)(x + __i + 12));                     \
        y_v_0 = vmulq_f32(x_v_0, val_v);                                \
        y_v_1 = vmulq_f32(x_v_1, val_v);                                \
        y_v_2 = vmulq_f32(x_v_2, val_v);                                \
        y_v_3 = vmulq_f32(x_v_3, val_v);                                \
        vst1q_f32(&(y)[__i], y_v_0);                                    \
        vst1q_f32(&(y)[__i + 4], y_v_1);                                \
        vst1q_f32(&(y)[__i + 8], y_v_2);                                \
        vst1q_f32(&(y)[__i + 12], y_v_3);                               \
    }                                                                   \
    for (; __i < len - 3; __i += 4)                                     \
    {                                                                   \
        x_v_0 = vld1q_f32((float *)(x + __i));                          \
        y_v_0 = vmulq_f32(x_v_0, val_v);                                \
        vst1q_f32(&(y)[__i], y_v_0);                                    \
    }                                                                   \
    for (; __i < len; __i++)                                            \
    {                                                                   \
        (y)[__i] = (x)[__i] * __val;                                    \
    }                                                                   \
}

#define VEC_FMA2_S(y, x, __val, len)                                    \
do                                                                      \
{                                                                       \
    float32x4_t y_v_0, x_v_0, y_v_1, x_v_1, y_v_2, x_v_2, y_v_3, x_v_3; \
    float32x4_t val_v;                                                  \
    val_v = vdupq_n_f32((__val));                                       \
    int32_t len16 = (len)-15;                                           \
    int32_t __i = 0;                                                    \
    for (; __i < len16; __i += 16)                                      \
    {                                                                   \
        y_v_0 = vld1q_f32((float *)(y + __i));                          \
        x_v_0 = vld1q_f32((float *)(x + __i));                          \
        y_v_1 = vld1q_f32((float *)(y + __i + 4));                      \
        x_v_1 = vld1q_f32((float *)(x + __i + 4));                      \
        y_v_2 = vld1q_f32((float *)(y + __i + 8));                      \
        x_v_2 = vld1q_f32((float *)(x + __i + 8));                      \
        y_v_3 = vld1q_f32((float *)(y + __i + 12));                     \
        x_v_3 = vld1q_f32((float *)(x + __i + 12));                     \
        y_v_0 = vfmaq_f32(y_v_0, x_v_0, val_v);                         \
        y_v_1 = vfmaq_f32(y_v_1, x_v_1, val_v);                         \
        y_v_2 = vfmaq_f32(y_v_2, x_v_2, val_v);                         \
        y_v_3 = vfmaq_f32(y_v_3, x_v_3, val_v);                         \
        vst1q_f32(&(y)[__i], y_v_0);                                    \
        vst1q_f32(&(y)[__i + 4], y_v_1);                                \
        vst1q_f32(&(y)[__i + 8], y_v_2);                                \
        vst1q_f32(&(y)[__i + 12], y_v_3);                               \
    }                                                                   \
    for (; __i < len - 3; __i += 4)                                     \
    {                                                                   \
        y_v_0 = vld1q_f32((float *)(y + __i));                          \
        x_v_0 = vld1q_f32((float *)(x + __i));                          \
        y_v_0 = vfmaq_f32(y_v_0, x_v_0, val_v);                         \
        vst1q_f32(&(y)[__i], y_v_0);                                    \
    }                                                                   \
    for (; __i < len; __i++)                                            \
    {                                                                   \
        (y)[__i] += (x)[__i] * __val;                                   \
    }                                                                   \
} while (0)

void mm_csr_ntile_ktile_mtile_unroll(const float alpha, const internal_spmat mat,
                                            const float *x, const ALPHA_INT columns,
                                            const ALPHA_INT ldx, const float beta, float *y,
                                            const ALPHA_INT ldy, ALPHA_INT lrs, ALPHA_INT lre,
                                            ALPHA_INT Mtile, ALPHA_INT Ntile, ALPHA_INT Ktile) {
  y = &y[index2(lrs, 0, ldy)];
  ALPHA_INT lrl = lre - lrs;
  ALPHA_INT bkl;
  ALPHA_INT ldp;
  ALPHA_INT *pos;
  csr_col_partition(mat, lrs, lre, Ktile, &pos, &bkl, &ldp);

  for (ALPHA_INT bcs = 0; bcs < columns; bcs += Ntile)
  {
    ALPHA_INT bcl = alpha_min(bcs + Ntile, columns) - bcs;
    ALPHA_INT bcl4 = bcl - 3;
    for (ALPHA_INT r = 0; r < lrl; r++)
    {
      float *Ytile = &y[index2(r, bcs, ldy)];
      VEC_MUL2_S(Ytile, Ytile, beta, bcl);
    }
    for (ALPHA_INT brs = 0; brs < lrl; brs += Mtile)
    {
      ALPHA_INT brl = alpha_min(lrl, brs + Mtile) - brs;
      ALPHA_INT *POS = &pos[index2(brs, 0, ldp)];
      for (ALPHA_INT bki = 0; bki < bkl; bki++)
      {
        for (ALPHA_INT r = 0; r < brl; r++)
        {
          float *Ytile = &y[index2(r + brs, bcs, ldy)];
          ALPHA_INT kis = POS[index2(r, bki, ldp)];
          ALPHA_INT kie = POS[index2(r, bki + 1, ldp)];
          ALPHA_INT kil = kie - kis;
          float *A_val = &((float*)mat->val_data)[kis];
          ALPHA_INT *A_col = &mat->col_data[kis];
          for (ALPHA_INT ki = 0; ki < kil; ki++)
          {
            ALPHA_INT col = A_col[ki];
            float val;
            val = alpha_mul(alpha, A_val[ki]);
            const float *Xtile = &x[index2(col, bcs, ldx)];
            VEC_FMA2_S(Ytile, Xtile, val, bcl);
          }
        }
      }
    }
  }
  alpha_free(pos);
}

template <typename J>
void mm_csr_ntile_ktile_mtile_unroll(const J alpha, const internal_spmat mat,
                                            const J *x, const ALPHA_INT columns,
                                            const ALPHA_INT ldx, const J beta, J *y,
                                            const ALPHA_INT ldy, ALPHA_INT lrs, ALPHA_INT lre,
                                            ALPHA_INT Mtile, ALPHA_INT Ntile, ALPHA_INT Ktile) {
  y = &y[index2(lrs, 0, ldy)];
  ALPHA_INT lrl = lre - lrs;
  ALPHA_INT bkl;
  ALPHA_INT ldp;
  ALPHA_INT *pos;
  csr_col_partition(mat, lrs, lre, Ktile, &pos, &bkl, &ldp);

  for (ALPHA_INT bcs = 0; bcs < columns; bcs += Ntile) {
    ALPHA_INT bcl = alpha_min(bcs + Ntile, columns) - bcs;
    for (ALPHA_INT r = 0; r < lrl; r++) {
      J *Ytile = &y[index2(r, bcs, ldy)];
      for (ALPHA_INT c = 0; c < bcl; c++) {
        Ytile[c] = alpha_mul(Ytile[c], beta);
      }
    }
    for (ALPHA_INT brs = 0; brs < lrl; brs += Mtile) {
      ALPHA_INT brl = alpha_min(lrl, brs + Mtile) - brs;
      ALPHA_INT *POS = &pos[index2(brs, 0, ldp)];
      for (ALPHA_INT bki = 0; bki < bkl; bki++) {
        for (ALPHA_INT r = 0; r < brl; r++) {
          J *Ytile = &y[index2(r + brs, bcs, ldy)];
          ALPHA_INT kis = POS[index2(r, bki, ldp)];
          ALPHA_INT kie = POS[index2(r, bki + 1, ldp)];
          ALPHA_INT kil = kie - kis;
          J *A_val = &((J*)mat->val_data)[kis];
          ALPHA_INT *A_col = &mat->col_data[kis];
          for (ALPHA_INT ki = 0; ki < kil; ki++) {
            ALPHA_INT col = A_col[ki];
            J val;
            val = alpha_mul(alpha, A_val[ki]);
            const J *Xtile = &x[index2(col, bcs, ldx)];

            for (ALPHA_INT c = 0; c < bcl; c++) {
              Ytile[c] = alpha_madd(Xtile[c], val, Ytile[c]);
            }
          }
        }
      }
    }
    alpha_free(pos);
  }
}

template <typename J>
static alphasparseStatus_t mm_csr_omp(const J alpha, const internal_spmat mat,
                                      const J *x, const ALPHA_INT columns, const ALPHA_INT ldx,
                                      const J beta, J *y, const ALPHA_INT ldy) {
  ALPHA_INT Ktile, Ntile, Mtile;
  if (std::is_same_v<J, float>)
  {
    Ntile = 512;
    Ktile = 256;
    Mtile = 512;
  }
  else
  {
    Ntile = 1024;
    Ktile = 256;
    Mtile = 512;
  }
  Mtile = mat->rows;
  Ktile = mat->cols;
  Ntile = columns;
  ALPHA_INT num_threads = alpha_get_thread_num();
  ALPHA_INT partition[num_threads + 1];
  balanced_partition_row_by_nnz(mat->row_data+1, mat->rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT lrs = partition[tid];
    ALPHA_INT lre = partition[tid + 1];
    mm_csr_ntile_ktile_mtile_unroll(alpha, mat, x, columns, ldx, beta, y, ldy, lrs, lre, Mtile,
                                    Ntile, Ktile);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename J>
alphasparseStatus_t gemm_csr_row(const J alpha, const internal_spmat mat, const J *x,
                          const ALPHA_INT columns, const ALPHA_INT ldx, const J beta,
                          J *y, const ALPHA_INT ldy) {
  return mm_csr_omp<J>(alpha, mat, x, columns, ldx, beta, y, ldy);
}
