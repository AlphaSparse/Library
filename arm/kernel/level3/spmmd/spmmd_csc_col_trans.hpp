#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename J>
alphasparseStatus_t spmmd_csc_col_trans(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    internal_spmat transposed_mat;
    transpose_csc<J>(matA, &transposed_mat);
    alphasparseStatus_t status = spmmd_csc_col<J>(transposed_mat,matB,matC,ldc);
    destroy_csc(transposed_mat);
    return status;
}
