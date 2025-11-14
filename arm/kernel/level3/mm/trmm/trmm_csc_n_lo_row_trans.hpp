#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename J>
alphasparseStatus_t trmm_csc_n_lo_row_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_csc<J>(mat, &transposed_mat);
    alphasparseStatus_t status = trmm_csc_n_hi_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csc(transposed_mat);
    return status;
}
