#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_coo_n_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT n = columns;
    ALPHA_INT _nnz = mat->nnz;
    ALPHA_INT con_or = 0;

    for (ALPHA_INT nnz = 0; nnz < _nnz; ++nnz)
    {
        ALPHA_INT r = mat->row_data[nnz];
        J *Y = &y[index2(r, 0, ldy)];
        while(con_or <= r)
        {
            J *TY = &y[index2(con_or, 0, ldy)];
            for (ALPHA_INT c = 0; c < n; c++)
                // Y[c] = Y[c] * beta;
                TY[c] = alpha_mul(TY[c], beta);

            con_or++;
        }

        if (mat->col_data[nnz] == r)
        {
            J val;
            val = alpha_mul(alpha, ((J *)mat->val_data)[nnz]);
            const J *X = &x[index2(mat->col_data[nnz], 0, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                Y[c] = alpha_madde(Y[c], val, X[c]);
                // Y[c] += val * X[c];
        }
    }

    while(con_or < mat->rows)
    {
        J *TY = &y[index2(con_or, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            // Y[c] = Y[c] * beta;
            TY[c] = alpha_mul(TY[c], beta);

        con_or++;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
