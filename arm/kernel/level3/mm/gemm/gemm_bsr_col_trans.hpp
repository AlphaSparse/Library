#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t gemm_bsr_col_trans(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat transposed_mat;
    transpose_bsr<TYPE>(mat, &transposed_mat);
    alphasparseStatus_t status = gemm_bsr_col(alpha, transposed_mat, x, columns, ldx, beta, y, ldy);
    destroy_bsr(transposed_mat);
    return status;
}
