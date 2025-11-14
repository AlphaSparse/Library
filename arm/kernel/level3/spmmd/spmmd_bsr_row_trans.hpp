#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename J>
alphasparseStatus_t spmmd_bsr_row_trans(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    internal_spmat transposed_mat;
    transpose_bsr<J>(matA, &transposed_mat);
    alphasparseStatus_t status = spmmd_bsr_row<J>(transposed_mat,matB,matC,ldc);
    destroy_bsr(transposed_mat);
    return status;
}
