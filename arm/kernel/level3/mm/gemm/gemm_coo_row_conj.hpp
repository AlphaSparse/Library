#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t gemm_coo_row_conj(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat conjugated_mat;
    transpose_conj_coo<TYPE>(mat, &conjugated_mat);
    alphasparseStatus_t status = gemm_coo_row(alpha, conjugated_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(conjugated_mat);
    return status;
}
