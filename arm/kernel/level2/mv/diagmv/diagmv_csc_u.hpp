#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
diagmv_csc_u(const TYPE alpha,
                     const internal_spmat A,
                     const TYPE *x,
                     const TYPE beta,
                     TYPE *y)
{                                                                          
    const int m = A->cols;

    for(int i = 0; i < m; ++i)
    {
        TYPE tmp;
        tmp = alpha_mul(alpha, x[i]);  
        y[i] = alpha_mul(y[i], beta);  
        y[i] = alpha_add(y[i], tmp);   
        // y[i] = beta * y[i] + alpha * x[i];
    }  
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
