#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_bsr_n_lo_row_conj(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<J>(A, &conjugated_mat);
    alphasparseStatus_t status = trsm_bsr_n_hi_row(alpha, conjugated_mat, x, columns, ldx, y, ldy);
    destroy_bsr(conjugated_mat);
    return status;
}
