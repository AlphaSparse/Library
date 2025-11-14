#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trmm_bsr_u_hi_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows * mat->block_dim;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_dim;

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT c = 0; c < n; ++c)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;
                for (ALPHA_INT lr = 0; lr < ll; ++lr)
                { // choose a inner row
                    J extra;
                    extra = alpha_setzero(extra);
                    for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
                    { // choose a block
                        ALPHA_INT ac = mat->col_data[ai] * ll;
                        J *blk = &((J *)mat->val_data)[ai * ll * ll];
                        
                        if (br == mat->col_data[ai])
                        {                                      // this is a diag block
                            extra = x[index2(c, r + lr, ldx)]; // this works only for hi-triangulars
                            for (ALPHA_INT lc = lr + 1; lc < ll; ++lc)
                            {
                                // extra += blk[index2(lr, lc, ll)] * x[index2(c, ac + lc, ldx)];
                                extra = alpha_madde(extra, blk[index2(lr, lc, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                        else if (br < mat->col_data[ai])
                        {
                            for (ALPHA_INT lc = 0; lc < ll; ++lc)
                            {
                                // extra += blk[index2(lr, lc, ll)] * x[index2(c, ac + lc, ldx)];
                                extra = alpha_madde(extra, blk[index2(lr, lc, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                    }
                    // y[index2(c, r + lr, ldy)] = beta * y[index2(c, r + lr, ldy)] + alpha * extra;
                    y[index2(c, r + lr, ldy)] = alpha_mul(beta, y[index2(c, r + lr, ldy)]);
                    y[index2(c, r + lr, ldy)] = alpha_madde(y[index2(c, r + lr, ldy)], alpha, extra);
                }
            }
        }
        break;
    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:    
        for (ALPHA_INT c = 0; c < n; ++c)
        { // choose a column from x
            for (ALPHA_INT r = 0; r < m; r += ll)
            { // choose a block of row
                ALPHA_INT br = r / ll;

                for (ALPHA_INT lr = 0; lr < ll; ++lr)
                { // choose a inner row
                    J extra;
                    extra = alpha_setzero(extra);
                    for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
                    { // choose a block
                        ALPHA_INT ac = mat->col_data[ai] * ll;
                        J *blk = &((J *)mat->val_data)[ai * ll * ll];

                        if (br == mat->col_data[ai])
                        { // this is a diag block
                            extra = x[index2(c, r + lr, ldx)];
                            for (ALPHA_INT lc = lr + 1; lc < ll; ++lc)
                            {
                                // extra += blk[index2(lc, lr, ll)] * x[index2(c, ac + lc, ldx)];
                                alpha_madde(extra, blk[index2(lc, lr, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                        else if (br < mat->col_data[ai])
                        {
                            for (ALPHA_INT lc = 0; lc < ll; ++lc)
                            {
                                // extra += blk[index2(lc, lr, ll)] * x[index2(c, ac + lc, ldx)];
                                extra = alpha_madde(extra, blk[index2(lc, lr, ll)], x[index2(c, ac + lc, ldx)]);
                            }
                        }
                    }
                    // y[index2(c, r + lr, ldy)] = beta * y[index2(c, r + lr, ldy)] + alpha * extra;
                    y[index2(c, r + lr, ldy)] = alpha_mul(beta, y[index2(c, r + lr, ldy)]);
                    y[index2(c, r + lr, ldy)] = alpha_madde(y[index2(c, r + lr, ldy)], alpha, extra);
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
