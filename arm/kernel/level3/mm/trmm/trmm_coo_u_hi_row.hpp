#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trmm_coo_u_hi_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT n = columns;
    ALPHA_INT r = 0;
    for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
    {
        ALPHA_INT cr = mat->row_data[nn];
        J *Y = &y[index2(cr, 0, ldy)];
        while(r <= cr)
        {
            J *TY = &y[index2(r, 0, ldy)];
            for (ALPHA_INT c = 0; c < n; c++)
            {
                TY[c] = alpha_mul(TY[c], beta);
                TY[c] = alpha_madde(TY[c], alpha, x[index2(r, c, ldx)]);
                // TY[c] = TY[c] * beta + alpha * x[index2(r, c, ldy)];
            }
            r++;
        }
        if(mat->col_data[nn] > cr)
        {
            J val;
            val = alpha_mul(alpha, ((J *)mat->val_data)[nn]);
            const J *X = &x[index2(mat->col_data[nn], 0, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                Y[c] = alpha_madde(Y[c], val, X[c]);
                // Y[c] += val * X[c];
        }
    }
    while(r < mat->rows)
    {
        J *TY = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
        {
            TY[c] = alpha_mul(TY[c], beta);
            // TY[c] = TY[c] * beta;
        }
        r++;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
