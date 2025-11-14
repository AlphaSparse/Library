#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_csc_u_hi_col_conj(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<J>(A, &conjugated_mat);
    alphasparseStatus_t status = trsm_csc_u_lo_col(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_csc(conjugated_mat);
    return status;
}
