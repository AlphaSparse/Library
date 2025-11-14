#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
symv_csc_u_hi(const TYPE alpha,
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
        TYPE tmp1, tmp2;
        tmp1 = alpha_mul(alpha, x[i]); 
        tmp2 = alpha_mul(beta, y[i]); 
        y[i] = alpha_add(tmp1, tmp2);
		// y[i] = beta * y[i] + alpha * x[i];
	}
	for(ALPHA_INT i = 0; i < n; ++i)
    {
        for(ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ++ai)
        {
            const ALPHA_INT row = A->row_data[ai];
            if(row >= i)
            {                                                                           
                continue;
            }
            else
            {   
                TYPE tmp;
                tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[i]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[row] = alpha_add(y[row], tmp);
				
				tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[row]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[i] = alpha_add(y[i], tmp);
 
                // y[col] += alpha * ((TYPE *)A->val_data)[ai] * x[i];
                // y[i] += alpha * ((TYPE *)A->val_data)[ai] * x[col];
            }
        }
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
