#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t gemm_dia_row(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT r = 0; r < mat->rows; ++r)
    {
        TYPE *Y = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < columns; c++)
            Y[c] = alpha_mul(Y[c],beta);
    }
    for(ALPHA_INT di = 0; di < mat->ndiag;++di){
        ALPHA_INT d = mat->dis_data[di];
        ALPHA_INT ars = alpha_max(0,-d);
        ALPHA_INT acs = alpha_max(0,d);
        ALPHA_INT an = alpha_min(mat->rows - ars,mat->cols - acs);
        for(ALPHA_INT i = 0; i < an; ++i){
            ALPHA_INT ar = ars + i;
            ALPHA_INT ac = acs + i;
            TYPE *Y = &y[index2(ar, 0, ldy)];
            const TYPE *X = &x[index2(ac, 0, ldx)];
            TYPE val;
            val = alpha_mul(((TYPE *)mat->val_data)[index2(di,ar,mat->lval)],alpha);
            for(ALPHA_INT bc = 0;bc < columns;++bc){
                Y[bc] = alpha_madde(Y[bc],val,X[bc]);
            }
        }
    } 	
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
