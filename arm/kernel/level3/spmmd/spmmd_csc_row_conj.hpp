#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename J>
alphasparseStatus_t spmmd_csc_row_conj(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<J>(matA, &conjugated_mat);
    alphasparseStatus_t status = spmmd_csc_row<J>(conjugated_mat,matB,matC,ldc);
    destroy_csc(conjugated_mat);
    return status;
}
