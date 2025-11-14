#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

template <typename J>
alphasparseStatus_t trmm_coo_u_hi_row_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_coo<J>(mat, &transposed_mat);
    alphasparseStatus_t status = trmm_coo_u_lo_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(transposed_mat);
    return status;
}
