#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_sky_n_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;

    for (ALPHA_INT r = 0; r < m; ++r)
    {
        J *Y = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            Y[c] = alpha_mul(Y[c], beta);

        ALPHA_INT idx = mat->pointers[r + 1] - 1;
        J val;
        val = alpha_mul(alpha, ((J *)mat->val_data)[idx]);
        const J *X = &x[index2(r, 0, ldx)];
        for (ALPHA_INT c = 0; c < n; ++c)
            Y[c] = alpha_madde(Y[c], val, X[c]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
