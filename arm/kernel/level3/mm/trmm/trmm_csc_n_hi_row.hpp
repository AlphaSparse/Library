#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trmm_csc_n_hi_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
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
            if(ar <= ac)
            {
                J val;// = alpha * ((J *)mat->val_data)[ai];
                const J *X = &x[index2(ac, 0, ldx)];
                J *Y = &y[index2(ar, 0, ldy)];
                
                val = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                for(ALPHA_INT cc = 0; cc < columns; ++cc)
                    Y[cc] = alpha_madde(Y[cc], val, X[cc]);//Y[cc] += val * X[cc];
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
