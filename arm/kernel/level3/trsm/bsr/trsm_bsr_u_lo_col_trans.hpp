#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_bsr_u_lo_col_trans(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_bsr<J>(A, &transposed_mat);
    alphasparseStatus_t status = trsm_bsr_u_hi_col(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_bsr(transposed_mat);
    return status;
}
