#include "alphasparse/kernel.h"
#include "format/transpose_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t spmmd_csr_col_trans(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    internal_spmat transposed_mat;
    transpose_csr<J>(matA, &transposed_mat);
    alphasparseStatus_t status = spmmd_csr_col(transposed_mat,matB,matC,ldc);
    destroy_csr(transposed_mat);
    return status;
}
