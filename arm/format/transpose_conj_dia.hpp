#ifndef TRANSPOSE_CONJ_DIA_HPP
#define TRANSPOSE_CONJ_DIA_HPP
#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

template <typename J>
alphasparseStatus_t transpose_conj_dia(const internal_spmat A, internal_spmat *B)
{
    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *B = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT colA = A->cols;
    ALPHA_INT ndiagA = A->ndiag;
    mat->rows = colA;
    mat->cols = rowA;
    mat->ndiag = ndiagA;
    mat->lval = mat->rows;
    mat->dis_data = (ALPHA_INT*)alpha_malloc((uint64_t)sizeof(ALPHA_INT)*ndiagA);
    for(ALPHA_INT i = 0;i<ndiagA;++i){
        mat->dis_data[i] = 0-A->dis_data[ndiagA - i -1];
    }
    mat->val_data = (J*)alpha_malloc((uint64_t)sizeof(J)*mat->ndiag*mat->lval);
    memset(mat->val_data,'\0',(uint64_t)sizeof(J)*mat->ndiag*mat->lval);
    for(ALPHA_INT adi = 0,bdi = ndiagA - 1;adi<ndiagA;++adi,--bdi){
        ALPHA_INT ad = A->dis_data[adi];
        ALPHA_INT bd = mat->dis_data[bdi];
        ALPHA_INT ars = alpha_max(0,-ad);
        ALPHA_INT brs = alpha_max(0,-bd);
        ALPHA_INT acs = alpha_max(0,ad);
        ALPHA_INT an = alpha_min(rowA - ars,colA - acs);
        for(ALPHA_INT j = 0;j<an;++j){
            ((J*)mat->val_data)[index2(bdi,brs+j,mat->lval)] = cmp_conj(((J*)A->val_data)[index2(adi,ars+j,A->lval)]);    
        }
    }
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}
#endif