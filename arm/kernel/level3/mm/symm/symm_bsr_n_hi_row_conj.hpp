#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_bsr_n_hi_row_conj(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows * mat->block_dim;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_dim;

    for (ALPHA_INT r = 0; r < m; ++r)
        for (ALPHA_INT c = 0; c < n; c++)
            // y[index2(r, c, ldy)] *= beta;
            y[index2(r, c, ldy)] = alpha_mul(beta, y[index2(r, c, ldy)]);

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT r = 0; r < m; r += ll)
        {
            ALPHA_INT br = r / ll;
            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
            {
                ALPHA_INT lr, lc;
                ALPHA_INT ac = mat->col_data[ai] * ll;
                J *blk = &((J *)mat->val_data)[ai * ll * ll];

                if (br == mat->col_data[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                    {
                        for (ALPHA_INT c = 0; c < n; ++c)
                        {
                            // y[index2(r + lr, c, ldy)] += alpha * blk[index2(lr, lr, ll)] * x[index2(ac+lr, c, ldx)];
                            J tmp;
                            tmp = alpha_mul_2c(blk[index2(lr, lr, ll)], x[index2(ac+lr, c, ldx)]);
                            y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], alpha, tmp);
                        }

                        for (lc = lr+1; lc < ll; ++lc)
                        {
                            J val;
                            val = alpha_conj(blk[index2(lr, lc, ll)]);
                            const J *X = &x[index2(ac + lc, 0, ldx)];
                            const J *Xsym = &x[index2(r + lr, 0, ldx)];
                            val = alpha_mul(alpha, val);

                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], val, X[c]);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(ac + lc, c, ldy)] += val * Xsym[c];
                                y[index2(ac + lc, c, ldy)] = alpha_madde(y[index2(ac + lc, c, ldy)], val, Xsym[c]);
                        } 
                    }
                }
                else if (br < mat->col_data[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                        for (lc = 0; lc < ll; ++lc)
                        {
                            J val;
                            val = alpha_conj(blk[index2(lr, lc, ll)]);
                            const J *X = &x[index2(ac + lc, 0, ldx)];
                            const J *Xsym = &x[index2(r + lr, 0, ldx)];
                            val = alpha_mul(alpha, val);

                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], val, X[c]);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(ac + lc, c, ldy)] += val * Xsym[c];
                                y[index2(ac + lc, c, ldy)] = alpha_madde(y[index2(ac + lc, c, ldy)], val, Xsym[c]);
                        }
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT r = 0; r < m; r += ll)
        {
            ALPHA_INT br = r / ll;
            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
            {
                ALPHA_INT lr, lc;
                ALPHA_INT ac = mat->col_data[ai] * ll;
                J *blk = &((J *)mat->val_data)[ai * ll * ll];

                if (br == mat->col_data[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                    {
                        for (ALPHA_INT c = 0; c < n; ++c)
                        {    
                            // y[index2(r + lr, c, ldy)] += alpha * blk[index2(lr, lr, ll)] * x[index2(ac+lr, c, ldx)];
                            J tmp;
                            tmp = alpha_mul_2c(blk[index2(lr, lr, ll)], x[index2(ac+lr, c, ldx)]);
                            y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], alpha, tmp);
                        }

                        for (lc = lr+1; lc < ll; ++lc)
                        {
                            J val;
                            val = alpha_conj(blk[index2(lc, lr, ll)]);
                            const J *X = &x[index2(ac + lc, 0, ldx)];
                            const J *Xsym = &x[index2(r + lr, 0, ldx)];
                            val = alpha_mul(alpha, val);

                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], val, X[c]);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(ac + lc, c, ldy)] += val * Xsym[c];
                                y[index2(ac + lc, c, ldy)] = alpha_madde(y[index2(ac + lc, c, ldy)], val, Xsym[c]);
                        } 
                    }
                }
                else if (br < mat->col_data[ai])
                {
                    for (lr = 0; lr < ll; ++lr)
                        for (lc = 0; lc < ll; ++lc)
                        {
                            J val;
                            val = alpha_conj(blk[index2(lc, lr, ll)]);
                            const J *X = &x[index2(ac + lc, 0, ldx)];
                            const J *Xsym = &x[index2(r + lr, 0, ldx)];
                            val = alpha_mul(alpha, val);

                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(r + lr, c, ldy)] += val * X[c];
                                y[index2(r + lr, c, ldy)] = alpha_madde(y[index2(r + lr, c, ldy)], val, X[c]);
                            for (ALPHA_INT c = 0; c < n; ++c)
                                // y[index2(ac + lc, c, ldy)] += val * Xsym[c];
                                y[index2(ac + lc, c, ldy)] = alpha_madde(y[index2(ac + lc, c, ldy)], val, Xsym[c]);
                        }
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
