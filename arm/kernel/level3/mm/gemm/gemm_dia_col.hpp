#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t gemm_dia_col(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    TYPE tmp;
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        TYPE *Y = &y[index2(cc, 0, ldy)];
        const TYPE *X = &x[index2(cc, 0, ldx)];
        for (ALPHA_INT r = 0; r < mat->rows; ++r)
            Y[r] = alpha_mul(Y[r],beta);
        for(ALPHA_INT di = 0; di < mat->ndiag;++di){
            ALPHA_INT d = mat->dis_data[di];
            ALPHA_INT ars = alpha_max(0,-d);
            ALPHA_INT acs = alpha_max(0,d);
            ALPHA_INT an = alpha_min(mat->rows - ars,mat->cols - acs);
            for(ALPHA_INT i = 0; i < an; ++i){
                ALPHA_INT ar = ars + i;
                ALPHA_INT ac = acs + i;
                tmp = alpha_mul(alpha,((TYPE *)mat->val_data)[index2(di,ar,mat->lval)]);
                Y[ar] = alpha_madde(Y[ar],tmp,X[ac]);
            }
        } 	
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
