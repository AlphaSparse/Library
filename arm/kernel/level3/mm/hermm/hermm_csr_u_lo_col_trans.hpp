#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t hermm_csr_u_lo_col_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_csr<J>(mat, &transposed_mat);
    for (ALPHA_INT r = 0; r < transposed_mat->rows; ++r){
        for (ALPHA_INT i = transposed_mat->row_data[r]; i < transposed_mat->row_data[r+1]; ++i){
            ALPHA_INT c = transposed_mat->col_data[i];
            if(r == c)
                ((J*)transposed_mat->val_data)[i] = cmp_conj(((J*)transposed_mat->val_data)[i]);
        }
    }
    alphasparseStatus_t status = hermm_csr_u_hi_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csr(transposed_mat);
    return status;
}
