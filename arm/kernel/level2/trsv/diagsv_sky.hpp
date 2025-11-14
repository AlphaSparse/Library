#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t diagsv_sky_n(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(TYPE));

    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
        diag[r - 1] = ((TYPE *)A->val_data)[indx];
    }
    
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        TYPE t;
        t = alpha_mul(alpha, x[r]);
        y[r] = alpha_div(t, diag[r]);
        // y[r] = alpha * x[r] / diag[r];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t diagsv_sky_u(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        y[r] = alpha_mul(alpha, x[r]);
        // y[r] = alpha * x[r];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}