#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_dia.hpp"
#include "format/transpose_conj_dia.hpp"
#include "format/destroy_dia.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_dia_n_lo_row_conj(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT r = 0; r < mat->rows; r++)
        for(ALPHA_INT c = 0; c < columns; c++)
            y[index2(r,c,ldy)] = alpha_mul(y[index2(r,c,ldy)],beta);
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
                for(ALPHA_INT bc = 0;bc < columns;++bc){
                    y[index2(ar,bc,ldy)] = alpha_madde(y[index2(ar,bc,ldy)],val,x[index2(ac,bc,ldx)]);
                    y[index2(ac,bc,ldy)] = alpha_madde(y[index2(ac,bc,ldy)],val,x[index2(ar,bc,ldx)]); 
                }
            }
        }
        if(d == 0){
            for(ALPHA_INT r = 0; r < mat->rows; ++r){
                J val;
                val = alpha_mul_2c(((J *)mat->val_data)[index2(di,r,mat->lval)],alpha);
                for(ALPHA_INT bc = 0;bc < columns;++bc){
                    y[index2(r,bc,ldy)] = alpha_madde(y[index2(r,bc,ldy)],val,x[index2(r,bc,ldx)]);
                }
            }
        }
    } 	
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
