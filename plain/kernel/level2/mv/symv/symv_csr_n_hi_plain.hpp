#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"

#include <memory.h>
#include <stdlib.h>

template <typename TYPE>
alphasparseStatus_t
symv_csr_n_hi_plain(const TYPE alpha, const internal_spmat A, const TYPE *x,
                          const TYPE beta, TYPE *y) {

    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
  
    for(ALPHA_INT i = 0; i < m; ++i)
    {
      y[i] = alpha_mul(y[i], beta);
    }
    for(ALPHA_INT i = 0; i < m; ++i)
    {
      for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
      {
        const ALPHA_INT col = A->col_data[ai];
        TYPE tmp;
        tmp = alpha_setzero(tmp);			
        if(col < i)
        {
          continue;
        }
        else if(col == i)
        {
          tmp = alpha_mul(alpha, ((TYPE*)A->val_data)[ai]);
          y[i] = alpha_madde(y[i], tmp, x[col]);
        }
        else
        {
          tmp = alpha_mul(alpha, ((TYPE*)A->val_data)[ai]);
          y[col] = alpha_madde(y[col], tmp, x[i]);
          y[i] = alpha_madde(y[i], tmp, x[col]);
        }
      }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
