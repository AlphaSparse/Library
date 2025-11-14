#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t gemm_bsr_col(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows * mat->block_dim;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_dim;

    for (ALPHA_INT j = 0; j < n; ++j)
        for (ALPHA_INT i = 0; i < m; ++i)
        {
            // y[index2(j, i, ldy)] *= beta;
            y[index2(j, i, ldy)] = alpha_mul(beta, y[index2(j, i, ldy)]);
        }

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT c = 0; c < n; c += ll)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;
                for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
                { // choose a block
                    TYPE *blk = &((TYPE *)mat->val_data)[ai * ll * ll];
                    for (ALPHA_INT cc = 0; cc < ll; ++cc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    { // choose a inner row

                        ALPHA_INT ac = mat->col_data[ai] * ll;
                        TYPE extra;
                        extra = alpha_setzero(extra);

                        for (ALPHA_INT lc = 0; lc < ll; ++lc)
                        {
                            // extra += blk[index2(lr, lc, ll)] * x[index2(c + cc, ac + lc, ldx)];
                            extra = alpha_madde(extra, blk[index2(lr, lc, ll)], x[index2(c + cc, ac + lc, ldx)]);
                        }
                        // y[index2(c + cc, r + lr, ldy)] += alpha * extra;
                        y[index2(c + cc, r + lr, ldy)] = alpha_madde(y[index2(c + cc, r + lr, ldy)], alpha, extra);
                    }
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT c = 0; c < n; c += ll)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;
                for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
                { // choose a block
                    for (ALPHA_INT cc = 0; cc < ll; ++cc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    { // choose a inner row

                        ALPHA_INT ac = mat->col_data[ai] * ll;
                        TYPE *blk = &((TYPE *)mat->val_data)[ai * ll * ll];
                        TYPE extra;
                        extra = alpha_setzero(extra);

                        for (ALPHA_INT lc = 0; lc < ll; ++lc)
                        {
                            // extra += blk[index2(lc, lr, ll)] * x[index2(c + cc, ac + lc, ldx)];
                            extra = alpha_madde(extra, blk[index2(lc, lr, ll)], x[index2(c + cc, ac + lc, ldx)]);
                        }
                        // y[index2(c + cc, r + lr, ldy)] += alpha * extra;
                        y[index2(c + cc, r + lr, ldy)] = alpha_madde(y[index2(c + cc, r + lr, ldy)], alpha, extra);
                    }
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
