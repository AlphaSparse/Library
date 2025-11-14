#include "alphasparse/kernel.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t trmm_csr_u_lo_row_conj(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    internal_spmat conjugated_mat;
    transpose_conj_csr<J>(mat, &conjugated_mat);
    alphasparseStatus_t status = trmm_csr_u_hi_row(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_csr(conjugated_mat);
    return status;
}
