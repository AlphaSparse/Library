#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t
trmv_csr_u_lo_conj(const TYPE alpha,
      const internal_spmat A,
      const TYPE *x,
      const TYPE beta,
      TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    for(ALPHA_INT i = 0; i < m; ++i)
    {
        y[i] = alpha_mul(y[i], beta);
		y[i] = alpha_madd(alpha, x[i], y[i]);
        for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
        {
            const ALPHA_INT col = A->col_data[ai];
            if(col < i)
            {
                TYPE tmp;
                tmp = cmp_conj(((TYPE *)A->val_data)[ai]);
                tmp = alpha_mul(alpha, tmp);
				y[col] = alpha_madd(tmp, x[i], y[col]);
            }
        }
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
