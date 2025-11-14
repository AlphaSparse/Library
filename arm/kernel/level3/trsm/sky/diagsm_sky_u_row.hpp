#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagsm_sky_u_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT r = 0; r < m; ++r)
    {
        for (ALPHA_INT c = 0; c < n; ++c)
        {
            y[index2(r, c, ldy)] = alpha_mul(alpha, x[index2(r, c, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
