#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename TYPE>
alphasparseStatus_t spmm_csc_trans(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat transposed_mat;
    transpose_csc<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = spmm_csc<TYPE>(transposed_mat, B, matC);
    destroy_csc(transposed_mat);
    return status;
}
