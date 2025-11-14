#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/destroy_csc.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_csc_u_hi_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT cc=0; cc<columns; ++cc)
    {
        const J *X = &x[index2(cc, 0, ldx)];
        J *Y = &y[index2(cc, 0, ldy)];
        for (ALPHA_INT r = 0; r < mat->rows; r++)
        {
            J tmp1, tmp2;
            tmp1 = alpha_mul(X[r], alpha);
            tmp2 = alpha_mul(Y[r], beta);
            Y[r] = alpha_add(tmp1, tmp2); 
        }

        for (ALPHA_INT br = 0; br < mat->cols; ++br)
        {
            //double xval = X[br];
            for (ALPHA_INT ai = mat->col_data[br]; ai < mat->col_data[br+1]; ++ai)
            {
                // ai @ [ar, br] & [br, ar]
                ALPHA_INT ar = mat->row_data[ai];
                J spval;
                J val_c;
                val_c = cmp_conj(((J *)mat->val_data)[ai]);
                val_c = alpha_mul(alpha, val_c);
                //double spval = alpha * ((J *)mat->val_data)[ai];
                spval = alpha_mul(alpha, ((J *)mat->val_data)[ai]);                
                if(ar < br) // non-diag element, deal with sym ele
                {
                    //double symval = X[ar];
                    //Y[ar] += spval * xval;
                    Y[ar] = alpha_madde(Y[ar], spval, X[br]);
                    //Y[br] += spval * symval;
                    Y[br] = alpha_madde(Y[br], val_c, X[ar]);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}