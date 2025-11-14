#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_dia.hpp"
#include "format/transpose_conj_dia.hpp"
#include "format/destroy_dia.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_dia_u_lo_col_conj(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{

    for(ALPHA_INT c = 0; c < columns; c++)
        for (ALPHA_INT r = 0; r < mat->rows; r++){
            y[index2(c,r,ldy)] = alpha_mul(y[index2(c,r,ldy)],beta);
            alpha_madde(y[index2(c,r,ldy)],x[index2(c,r,ldx)],alpha);
        }
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        J* Y = &y[index2(cc,0,ldy)];
        const J* X = &x[index2(cc,0,ldx)];
        for(ALPHA_INT di = 0; di < mat->ndiag;++di){
            ALPHA_INT d = mat->dis_data[di];
            if(d < 0){
                ALPHA_INT ars = alpha_max(0,-d);
                ALPHA_INT acs = alpha_max(0,d);
                ALPHA_INT an = alpha_min(mat->rows - ars,mat->cols - acs);
                for(ALPHA_INT i = 0; i < an; ++i){
                    ALPHA_INT ar = ars + i;
                    ALPHA_INT ac = acs + i;
                    J val;
                    val = alpha_mul_2c(((J *)mat->val_data)[index2(di,ar,mat->lval)],alpha);
                    Y[ar] = alpha_madde(Y[ar],val,X[ac]);
                    Y[ac] = alpha_madde(Y[ac],val,X[ar]);
                }
            }
        } 	
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
