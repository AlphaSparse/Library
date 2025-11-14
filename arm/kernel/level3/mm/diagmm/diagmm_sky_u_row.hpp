#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_sky_u_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;

    for (ALPHA_INT r = 0; r < rowC; ++r)
    {
        for (ALPHA_INT c = 0; c < colC; ++c)
        {
            J t;
            t = alpha_mul(alpha, x[index2(r, c, ldx)]);
            y[index2(r, c, ldy)] = alpha_mul(beta, y[index2(r, c, ldy)]);
            y[index2(r, c, ldy)] = alpha_add(y[index2(r, c, ldy)], t);
            // y[index2(r,c,ldy)] = beta * y[index2(r,c,ldy)] + alpha * x[index2(r,c,ldy)];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
