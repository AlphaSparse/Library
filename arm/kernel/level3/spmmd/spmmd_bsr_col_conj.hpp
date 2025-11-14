#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename J>
alphasparseStatus_t spmmd_bsr_col_conj(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<J>(matA, &conjugated_mat);
    alphasparseStatus_t status = spmmd_bsr_col<J>(conjugated_mat,matB,matC,ldc);
    destroy_bsr(conjugated_mat);
    return status;
}
