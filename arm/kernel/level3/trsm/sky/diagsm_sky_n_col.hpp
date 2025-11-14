#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagsm_sky_n_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    J diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(J));

    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        const ALPHA_INT indx = A->pointers[r + 1] - 1;
        diag[r] = ((J*)A->val_data)[indx];
    }

    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows; ++r)
        {
            J t;
            t = alpha_mul(alpha, x[index2(c, r, ldx)]);
            y[index2(c, r, ldy)] = alpha_div(t, diag[r]);
            // y[index2(c, r, ldy)] = alpha * x[index2(c, r, ldx)] / diag[r];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
