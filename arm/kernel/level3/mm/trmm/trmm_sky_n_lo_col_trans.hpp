#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"


template <typename J>
alphasparseStatus_t trmm_sky_n_lo_col_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    // internal_spmat transposed_mat;
    // transpose_sky<J>(mat, &transposed_mat);
    // alphasparseStatus_t status = trmm_sky_n_hi_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    // destroy_sky(transposed_mat);
    alphasparseStatus_t status = trmm_sky_n_hi_col(alpha, mat, x, columns, ldx, beta, y, ldy);
    return status;
}
