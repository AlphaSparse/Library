#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_coo_n_hi_col_trans(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_coo<J>(A, &transposed_mat);
    alphasparseStatus_t status = trsm_coo_n_lo_col(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_coo(transposed_mat);
    return status;
}
