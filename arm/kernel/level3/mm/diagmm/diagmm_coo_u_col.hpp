#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_coo_u_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;

    for (ALPHA_INT c = 0; c < colC; ++c)
    {
        for (ALPHA_INT r = 0; r < rowC; ++r)
        {
            J t;
            t = alpha_setzero(t);
            t = alpha_mul(alpha, x[index2(c, r, ldx)]);
            y[index2(c, r, ldy)] = alpha_mul(beta, y[index2(c, r, ldy)]);
            y[index2(c, r, ldy)] = alpha_add(y[index2(c, r, ldy)], t);
            // y[index2(c, r, ldy)] = beta * y[index2(c, r, ldy)] + alpha * x[index2(c, r, ldx)];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
