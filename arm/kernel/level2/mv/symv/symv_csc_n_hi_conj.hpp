#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t
symv_csc_n_hi_conj(const TYPE alpha,
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
        // y[i] *= beta;
        y[i] = alpha_mul(y[i], beta); 
    }
    for(ALPHA_INT i = 0; i < n; ++i)
    {
        for(ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ++ai)
        {
            TYPE tmp;
            tmp = alpha_setzero(tmp);  
            const ALPHA_INT row = A->row_data[ai];
            if(row > i)
            {
                continue;
            }
            else if(row == i)
            {
                tmp = cmp_conj(((TYPE *)A->val_data)[ai]);
                tmp = alpha_mul(tmp, x[row]);  
                tmp = alpha_mul(alpha, tmp); 
                y[i] = alpha_add(y[i], tmp);
                // y[i] += alpha * ((TYPE *)A->val_data)[ai] * x[row];
            }
            else
            {
                tmp = cmp_conj(((TYPE *)A->val_data)[ai]);
                tmp = alpha_mul(tmp, x[i]); 
                tmp = alpha_mul(alpha, tmp); 
                y[row] = alpha_add(y[row], tmp);
                // y[row] += alpha * ((TYPE *)A->val_data)[ai] * x[i];
                tmp = cmp_conj(((TYPE *)A->val_data)[ai]);
                tmp = alpha_mul(tmp, x[row]);  
                tmp = alpha_mul(alpha, tmp); 
                y[i] = alpha_add(y[i], tmp);
                // y[i] += alpha * ((TYPE *)A->val_data)[ai] * x[row];
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
