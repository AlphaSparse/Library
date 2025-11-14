#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t trsm_csr_u_hi_col_trans(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_csr<J>(A, &transposed_mat);
    alphasparseStatus_t status = trsm_csr_u_lo_col(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_csr(transposed_mat);
    return status;
}
