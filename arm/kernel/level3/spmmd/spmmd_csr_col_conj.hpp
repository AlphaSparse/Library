#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t spmmd_csr_col_conj(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    internal_spmat conjugated_mat;
    transpose_conj_csr<J>(matA, &conjugated_mat);
    alphasparseStatus_t status = spmmd_csr_col<J>(conjugated_mat,matB,matC,ldc);
    destroy_csr(conjugated_mat);
    return status;
}
