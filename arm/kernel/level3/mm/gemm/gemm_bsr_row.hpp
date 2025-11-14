#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t gemm_bsr_row(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows * mat->block_dim;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_dim;

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT r = 0; r < m; r += ll)
        {
            ALPHA_INT br = r / ll;
            for (ALPHA_INT lr = 0; lr < ll; ++lr)
                for (ALPHA_INT c = 0; c < n; c++)
                    // y[index2(r + lr, c, ldy)] *= beta;
                    y[index2(r + lr, c, ldy)] = alpha_mul(beta, y[index2(r + lr, c, ldy)]);

            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
            {
                ALPHA_INT lr, lc;
                ALPHA_INT ac = mat->col_data[ai] * ll;
                TYPE *blk = &((TYPE *)mat->val_data)[ai * ll * ll];

                for (lr = 0; lr < ll; ++lr)
                    for (lc = 0; lc < ll; ++lc)
                    {
                        TYPE val = blk[index2(lr, lc, ll)];
                        const TYPE *X = &x[index2(ac + lc, 0, ldx)];
                        val = alpha_mul(alpha, val);
                        for (ALPHA_INT c = 0; c < n; ++c)
                        {
                            // y[index2(r + lr, c, ldy)] += val * X[c];
                            y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], val, X[c]);
                        }
                    }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT r = 0; r < m; r += ll)
        {
            ALPHA_INT br = r / ll;
            for (ALPHA_INT lr = 0; lr < ll; ++lr)
                for (ALPHA_INT c = 0; c < n; c++)
                    // y[index2(r + lr, c, ldy)] *= beta;
                    y[index2(r + lr, c, ldy)] = alpha_mul(beta, y[index2(r + lr, c, ldy)]);

            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
            {
                ALPHA_INT lr, lc;
                ALPHA_INT ac = mat->col_data[ai] * ll;
                TYPE *blk = &((TYPE *)mat->val_data)[ai * ll * ll];

                for (lr = 0; lr < ll; ++lr)
                    for (lc = 0; lc < ll; ++lc)
                    {
                        TYPE val = blk[index2(lc, lr, ll)];
                        const TYPE *X = &x[index2(ac + lc, 0, ldx)];
                        val = alpha_mul(alpha, val);
                        for (ALPHA_INT c = 0; c < n; ++c)
                        {   
                            // y[index2(r + lr, c, ldy)] += val * X[c];
                            y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], val, X[c]);
                        }
                    }
            }
        }
        break;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
