#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/destroy_csc.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_csc_n_lo_row_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_csc<J>(mat, &transposed_mat);
    for (ALPHA_INT c = 0; c < transposed_mat->cols; ++c){
        for (ALPHA_INT i = transposed_mat->col_data[c]; i < transposed_mat->col_data[c+1]; ++i){
            ALPHA_INT r = transposed_mat->row_data[i];
            if(r == c)
                ((J *)transposed_mat->val_data)[i] = cmp_conj(((J *)transposed_mat->val_data)[i]);
        }
    }
    alphasparseStatus_t status = hermm_csc_n_hi_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csc(transposed_mat);
    return status;
}