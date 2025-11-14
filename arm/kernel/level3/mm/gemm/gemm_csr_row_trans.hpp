#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename TYPE>
alphasparseStatus_t gemm_csr_row_trans(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_csr<TYPE>(mat, &transposed_mat);
    alphasparseStatus_t status = gemm_csr_row(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_csr(transposed_mat);
    return status;
}
