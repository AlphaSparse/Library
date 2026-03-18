#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t gemm_coo_row_plain(const TYPE alpha, const internal_spmat mat, const TYPE *x,
                          const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta,
                          TYPE *y, const ALPHA_INT ldy) {
    ALPHA_INT n = columns;
    ALPHA_INT r = 0;//mat->row_indx[0];

    for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
    {
        ALPHA_INT cr = mat->row_data[nn];
        TYPE *Y = &y[index2(cr, 0, ldy)];
        while(cr >= r)
        {
            TYPE *TY = &y[index2(r, 0, ldy)];
            for (ALPHA_INT c = 0; c < n; c++)
                TY[c] = alpha_mul(TY[c], beta);

            r++;
        }

        TYPE val;
        val = alpha_mul(alpha, ((TYPE*)mat->val_data)[nn]);
        const TYPE *X = &x[index2(mat->col_data[nn], 0, ldx)];
        for (ALPHA_INT c = 0; c < n; ++c)
            Y[c] = alpha_madde(Y[c], val, X[c]);
            // Y[c] += val * X[c];
    }

    while(mat->rows > r)
    {
      TYPE *TY = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            TY[c] = alpha_mul(TY[c], beta);

        r++;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
