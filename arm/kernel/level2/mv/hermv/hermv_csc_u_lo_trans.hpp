#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/destroy_csc.hpp"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
hermv_csc_u_lo_trans(const TYPE alpha,
      const internal_spmat A,
      const TYPE *x,
      const TYPE beta,
      TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;    

    for(ALPHA_INT i = 0; i < n; ++i)
    {
        // y[i] = beta * y[i] + alpha * x[i];
        TYPE tmp1, tmp2;
        tmp1 = alpha_mul(alpha, x[i]); 
        tmp2 = alpha_mul(beta, y[i]); 
        y[i] = alpha_add(tmp1, tmp2);
    }
    
    for(ALPHA_INT i = 0; i < n; ++i)
    {
        for(ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ++ai)
        {
            const ALPHA_INT row = A->row_data[ai];
            if(row > i)
            {                
                TYPE tmp;
                TYPE conval;
				conval = cmp_conj(((TYPE *)A->val_data)[ai]);

                tmp = alpha_mul(conval, x[i]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[row] = alpha_add(y[row], tmp);
				
				tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[row]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[i] = alpha_add(y[i], tmp);                                                            
                // y[row] += alpha * ((TYPE *)A->val_data)[ai] * x[i];
                // y[i] += alpha * ((TYPE *)A->val_data)[ai] * x[row];
            }                                                                           
        }
    }

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
