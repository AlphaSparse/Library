#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
trmv_csc_n_hi_conj(const TYPE alpha,
				const internal_spmat A,
				const TYPE *x,
				const TYPE beta,
				TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    for(ALPHA_INT i = 0; i < n; ++i)
    {
        TYPE tmp;
        tmp = alpha_setzero(tmp); 
        for(ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ++ai)
        {
            const ALPHA_INT row = A->row_data[ai];
            if(row <= i)
            {
                TYPE tmp1;
                tmp1 = alpha_conj(((TYPE *)A->val_data)[ai]);
                tmp1 = alpha_mul(tmp1, x[row]); 
                tmp = alpha_add(tmp1, tmp);
                // tmp += ((TYPE *)A->val_data)[ai] * x[row];                                          
            }
        }
        tmp = alpha_mul(tmp, alpha); 
        TYPE tmp1;
        tmp1 = alpha_mul(beta, y[i]); 
        y[i] = alpha_add(tmp1, tmp);
        // y[i] = alpha * tmp + beta * y[i];
    }

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
