#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_dia.hpp"
#include "format/transpose_conj_dia.hpp"
#include "format/destroy_dia.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_dia_n_lo_row_trans(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_dia<J>(A, &transposed_mat);
    alphasparseStatus_t status = trsm_dia_n_hi_row(alpha, transposed_mat, x, columns, ldx, y, ldy);
    destroy_dia(transposed_mat);
    return status;
}
