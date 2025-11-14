#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_dia_u_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT r = 0; r < mat->rows; ++r)
    {
        for (ALPHA_INT c = 0; c < columns; ++c)
        {
            y[index2(r,c,ldy)] = alpha_mul(y[index2(r,c,ldy)],beta);
            y[index2(r,c,ldy)] = alpha_madde(y[index2(r,c,ldy)],x[index2(r,c,ldx)],alpha);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
