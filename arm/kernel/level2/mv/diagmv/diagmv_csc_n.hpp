#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

#include <stdio.h>
template <typename TYPE>
alphasparseStatus_t
diagmv_csc_n(const TYPE alpha,
                     const internal_spmat A,
                     const TYPE *x,
                     const TYPE beta,
                     TYPE *y)
{
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    ALPHA_INT flag = 1;
    for(ALPHA_INT i = 0; i < n; ++i)
    {
        TYPE t;
        t = alpha_mul(y[i], beta);
        if(flag) 
        {
            y[i] = t;
        }
        for(ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ++ai)
        {
            const ALPHA_INT row = A->row_data[ai];
            if(i == row)
            {
                TYPE tmp;
                if(flag)
                {
                    tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[row]);     
                    tmp = alpha_mul(alpha, tmp);   
                    y[i] = alpha_add(y[i], tmp);  
                    // y[i] += alpha * ((TYPE *)A->val_data)[ai] * x[col];
                    flag = 0;
                }
                else
                {
                    tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[row]);     
                    tmp = alpha_mul(alpha, tmp);  
                    y[i] = alpha_add(t, tmp);  
                    // y[i] = alpha * ((TYPE *)A->val_data)[ai] * x[col] + t;
                }
                break;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
