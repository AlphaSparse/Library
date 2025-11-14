#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagmm_bsr_u_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowC = mat->rows * mat->block_dim;
    ALPHA_INT colC = columns;

    for (ALPHA_INT r = 0; r < rowC; ++r) //遍历的时候和col优先相反
    {
        for (ALPHA_INT c = 0; c < colC; ++c)
        {
            //y[index2(r,c,ldy)] = beta * y[index2(r,c,ldy)] + alpha * x[index2(r,c,ldy)];
            J t1, t2;
            t1 = alpha_mul(beta, y[index2(r, c, ldy)]);
            t2 = alpha_mul(alpha, x[index2(r, c, ldy)]);
            y[index2(r, c, ldy)] = alpha_add(t1, t2);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
