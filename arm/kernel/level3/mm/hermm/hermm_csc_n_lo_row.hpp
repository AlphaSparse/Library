#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/destroy_csc.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_csc_n_lo_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = mat->cols;

    //prepare all y with beta
    for(ALPHA_INT j=0; j < m; ++j)
    for(ALPHA_INT i=0; i < columns; ++i)    
    {
        y[index2(j,i,ldy)] = alpha_mul(beta, y[index2(j,i,ldy)]);//y[i] *= beta;
    }

    for(ALPHA_INT ac = 0; ac<n; ++ac)
    { 
        for(ALPHA_INT ai = mat->col_data[ac]; ai < mat->col_data[ac+1]; ++ai)
        {
            ALPHA_INT ar = mat->row_data[ai];
            J val;
            J val_c;
            val_c = cmp_conj(((J *)mat->val_data)[ai]);
            val_c = alpha_mul(alpha, val_c);
            val = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
            //double val = alpha * ((J *)mat->val_data)[ai];

            if(ac < ar)
            { // val @ [ar, ac] & [ac, ar]
                for(ALPHA_INT cc = 0; cc < columns; ++cc)
                {
                    //y[index2(ar, cc, ldy)] += val * x[index2(ac, cc, ldx)];
                    y[index2(ar, cc, ldy)] = alpha_madde(y[index2(ar, cc, ldy)], val, x[index2(ac, cc, ldx)]);
                    //y[index2(ac, cc, ldy)] += val * x[index2(ar, cc, ldx)];
                    y[index2(ac, cc, ldy)] = alpha_madde(y[index2(ac, cc, ldy)], val_c, x[index2(ar, cc, ldx)]);
                }
            }
            else if(ac == ar)
            {
                for(ALPHA_INT cc = 0; cc < columns; ++cc)
                {
                    //y[index2(ar, cc, ldy)] += val * x[index2(ac, cc, ldx)];
                    y[index2(ar, cc, ldy)] = alpha_madde(y[index2(ar, cc, ldy)], val, x[index2(ac, cc, ldx)]);
                }
                
            }

        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}