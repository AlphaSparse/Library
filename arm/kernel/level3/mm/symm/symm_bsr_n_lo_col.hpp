#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_bsr_n_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows * mat->block_dim;
    ALPHA_INT n = columns;
    ALPHA_INT ll = mat->block_dim;

    for (ALPHA_INT c = 0; c < n; c++)
        for (ALPHA_INT r = 0; r < m; ++r)
            // y[index2(c, r, ldy)] *= beta;
            y[index2(c, r, ldy)] = alpha_mul(beta, y[index2(c, r, ldy)]);

    switch (mat->block_layout)
    {
    case ALPHA_SPARSE_LAYOUT_ROW_MAJOR:
        for (ALPHA_INT matC = 0; matC < n; matC += ll)
        for (ALPHA_INT R = 0; R < m; R += ll)
        {
            ALPHA_INT br = R / ll;
            
            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
            {
                ALPHA_INT ac = mat->col_data[ai] * ll;
                J *blk = &((J *)mat->val_data)[ai*ll*ll];
                
                if(br == mat->col_data[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        J extra;
                        extra = alpha_setzero(extra);
                        for (ALPHA_INT i=0; i < lr; ++i)
                        {
                            // extra += blk[index2(lr, i, ll)] * x[index2(matC+lc, ac+i, ldx)];
                            extra = alpha_madde(extra, blk[index2(lr, i, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        for (ALPHA_INT i=lr; i < ll; ++i)
                        {
                            // extra += blk[index2(i, lr, ll)] * x[index2(matC+lc, ac+i, ldx)];
                            extra = alpha_madde(extra, blk[index2(i, lr, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        // y[index2(matC+lc, R+lr, ldy)] += alpha * extra;
                        y[index2(matC+lc, R+lr, ldy)] = alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                    }
                }
                else if(br > mat->col_data[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        J extra,extra_sym;
                        extra = alpha_setzero(extra);
                        extra_sym = alpha_setzero(extra_sym);
                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            // extra += blk[index2(lr, i, ll)] * x[index2(matC+lc, ac+i, ldx)];
                            extra = alpha_madde(extra, blk[index2(lr, i, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }

                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            // extra_sym += blk[index2(i, lr, ll)] * x[index2(matC+lc, R+i, ldx)];
                            extra_sym = alpha_madde(extra_sym, blk[index2(i, lr, ll)], x[index2(matC+lc, R+i, ldx)]);
                        }                        
                        
                        // y[index2(matC+lc, R+lr, ldy)] += alpha * extra;
                        y[index2(matC+lc, R+lr, ldy)] = alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                        // y[index2(matC+lc, ac+lr, ldy)] += alpha * extra_sym;
                        y[index2(matC+lc, ac+lr, ldy)] = alpha_madde(y[index2(matC+lc, ac+lr, ldy)], alpha, extra_sym);
                    }
                }
            }
        }
        break;

    case ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR:
        for (ALPHA_INT matC = 0; matC < n; matC += ll)
        for (ALPHA_INT R = 0; R < m; R += ll)
        {
            ALPHA_INT br = R / ll;
            
            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai)
            {
                ALPHA_INT ac = mat->col_data[ai] * ll;
                J *blk = &((J *)mat->val_data)[ai*ll*ll];
                
                if(br == mat->col_data[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        J extra;
                        extra = alpha_setzero(extra);
                        for (ALPHA_INT i=0; i < lr; ++i)
                        {
                            // extra += blk[index2(i, lr, ll)] * x[index2(matC+lc, ac+i, ldx)];
                            extra = alpha_madde(extra, blk[index2(i, lr, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        for (ALPHA_INT i=lr; i < ll; ++i)
                        {
                            // extra += blk[index2(lr, i, ll)] * x[index2(matC+lc, ac+i, ldx)];
                            extra = alpha_madde(extra, blk[index2(lr, i, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }
                        
                        // y[index2(matC+lc, R+lr, ldy)] += alpha * extra;
                        y[index2(matC+lc, R+lr, ldy)] = alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                    }
                }
                else if(br > mat->col_data[ai])
                {
                    for (ALPHA_INT lc = 0; lc < ll; ++lc)
                    for (ALPHA_INT lr = 0; lr < ll; ++lr)
                    {
                        J extra,extra_sym;
                        extra = alpha_setzero(extra);
                        extra_sym = alpha_setzero(extra_sym);
                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            // extra += blk[index2(i, lr, ll)] * x[index2(matC+lc, ac+i, ldx)];
                            extra = alpha_madde(extra, blk[index2(i, lr, ll)], x[index2(matC+lc, ac+i, ldx)]);
                        }

                        for (ALPHA_INT i = 0; i < ll; ++i)
                        {
                            // extra_sym += blk[index2(lr, i, ll)] * x[index2(matC+lc, R+i, ldx)];
                            extra_sym = alpha_madde(extra_sym, blk[index2(lr, i, ll)], x[index2(matC+lc, R+i, ldx)]);
                        }                        
                        
                        // y[index2(matC+lc, R+lr, ldy)] += alpha * extra;
                        y[index2(matC+lc, R+lr, ldy)] = alpha_madde(y[index2(matC+lc, R+lr, ldy)], alpha, extra);
                        // y[index2(matC+lc, ac+lr, ldy)] += alpha * extra_sym;
                        y[index2(matC+lc, ac+lr, ldy)] = alpha_madde(y[index2(matC+lc, ac+lr, ldy)], alpha, extra_sym);
                    }
                }
            }
        }
        break;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


