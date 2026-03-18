#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/bisearch.h"

template <typename J>
alphasparseStatus_t
symm_csr_n_hi_row_plain(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT r = 0; r < mat->rows; ++r)
      for (ALPHA_INT c = 0; c < columns; c++)
          y[index2(r, c, ldy)] = alpha_mul(y[index2(r, c, ldy)], beta);

    for (ALPHA_INT r = 0; r < mat->rows; ++r)
    {
      for (ALPHA_INT ai = mat->row_data[r]; ai < mat->row_data[r+1]; ai++)
      {
          ALPHA_INT ac = mat->col_data[ai];
          if (ac > r)
          {
              J val;
              val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
              for (ALPHA_INT c = 0; c < columns; ++c)
              {
                  y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                  y[index2(ac, c, ldy)] = alpha_madde(y[index2(ac, c, ldy)], val, x[index2(r, c, ldx)]);
              }
          }
          else if (ac == r)
          {
              J val;
              val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
              for (ALPHA_INT c = 0; c < columns; ++c)
              {
                  y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
              }
          }
      }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
