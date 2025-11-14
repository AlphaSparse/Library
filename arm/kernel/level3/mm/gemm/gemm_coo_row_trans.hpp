#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "format/transpose_coo.hpp"
#include "format/destroy_coo.hpp"

template <typename TYPE>
alphasparseStatus_t gemm_coo_row_trans(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_coo<TYPE>(mat, &transposed_mat);
    alphasparseStatus_t status = gemm_coo_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_coo(transposed_mat);
    return status;
}
