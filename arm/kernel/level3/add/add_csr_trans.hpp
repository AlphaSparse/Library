#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename TYPE>
alphasparseStatus_t add_csr_trans(const internal_spmat A, const TYPE alpha, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat transposed_mat;
    transpose_csr<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = add_csr<TYPE>(transposed_mat, alpha, B, matC);
    destroy_csr(transposed_mat);
    return status;
}
