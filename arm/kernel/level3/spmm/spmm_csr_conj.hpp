#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename TYPE>
alphasparseStatus_t spmm_csr_conj(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat conjugated_mat;
    transpose_conj_csr<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = spmm_csr<TYPE>(conjugated_mat, B, matC);
    destroy_csr(conjugated_mat);
    return status;
}