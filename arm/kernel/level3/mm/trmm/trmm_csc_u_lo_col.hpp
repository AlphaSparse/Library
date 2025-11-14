#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trmm_csc_u_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        const J *X = &x[index2(cc, 0, ldx)];
        J *Y = &y[index2(cc, 0, ldy)];
        for (ALPHA_INT r = 0; r < mat->rows; r++)
        {
            J tmp1, tmp2;
            tmp1 = alpha_mul(alpha, X[r]);
            tmp2 = alpha_mul(beta, Y[r]);
            Y[r] = alpha_add(tmp1, tmp2);
        }
        for(ALPHA_INT br = 0; br < mat->cols; ++br)
        {
            J xval = X[br];
            for (ALPHA_INT ai = mat->col_data[br]; ai < mat->col_data[br+1]; ++ai)
            {
                ALPHA_INT ar = mat->row_data[ai];
                if(br < ar)
                {
                    J spval;// = alpha * ((J *)mat->val_data)[ai];
                    spval = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                    Y[ar] = alpha_madde(Y[ar], spval, xval);
                    //Y[ar] += spval * xval; 
                }    
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
