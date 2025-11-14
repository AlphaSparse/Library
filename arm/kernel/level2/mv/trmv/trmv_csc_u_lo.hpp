#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
trmv_csc_u_lo(const TYPE alpha,
		              const internal_spmat A,
		              const TYPE *x,
		              const TYPE beta,
		              TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    TYPE tmp;

    for(ALPHA_INT i = 0; i < n; ++i)
	{
        tmp = alpha_mul(alpha, x[i]);
        y[i] = alpha_mul(beta, y[i]);
        y[i] = alpha_add(y[i], tmp);
		// y[i] = beta * y[i] + alpha * x[i];
	}
	for(ALPHA_INT i = 0; i < m; ++i)
    {
        for(ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ++ai)
        {
            const ALPHA_INT row = A->row_data[ai];
            if(row <= i)
            {
                continue;
            }
            else
            {
                tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[i]); 
				tmp = alpha_mul(alpha, tmp);
        		y[row] = alpha_add(tmp, y[row]);
                // y[row] += alpha * ((TYPE *)A->val_data)[ai] * x[i];
            }
        }
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
