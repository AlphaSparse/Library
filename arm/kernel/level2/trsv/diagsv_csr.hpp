#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t diagsv_csr_n(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(TYPE));

    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
        {
            ALPHA_INT ac = A->col_data[ai];
            if (ac == r)
            {
                diag[r] = ((TYPE *)A->val_data)[ai];
            }
        }
    }
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        y[r] = alpha_mul(alpha, x[r]);
        y[r] = alpha_div(y[r], diag[r]);
        // y[r] = alpha * x[r] / diag[r];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t diagsv_csr_u(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        y[r] = alpha_mul(alpha, x[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
