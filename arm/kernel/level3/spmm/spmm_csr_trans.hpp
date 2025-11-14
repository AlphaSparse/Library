#include "alphasparse/kernel.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t spmm_csr_trans(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat transposed_mat;
    transpose_csr<J>(A, &transposed_mat);
    alphasparseStatus_t status = spmm_csr<J>(transposed_mat, B, matC);
    destroy_csr(transposed_mat);
    return status;
}
