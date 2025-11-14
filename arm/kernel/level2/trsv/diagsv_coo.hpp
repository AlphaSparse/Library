#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t diagsv_coo_n(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        for (ALPHA_INT ai = 0; ai < A->nnz; ai++)
        {
            ALPHA_INT ar = A->row_data[ai];
            ALPHA_INT ac = A->col_data[ai];
            if (ac == r && ar == r)
            {
                TYPE t;
                t = alpha_mul(alpha, x[r]);
                y[r] = alpha_div(t, ((TYPE *)A->val_data)[ai]);
                // y[r] = (alpha * x[r]) / ((TYPE *)A->val_data)[ai];
                break;
            }
        }        
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t diagsv_coo_u(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        y[r] = alpha_mul(alpha, x[r]);
        // y[r] = alpha * x[r];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
