#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include <stdio.h>
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename TYPE>
alphasparseStatus_t spmm_csc_conj(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = spmm_csc<TYPE>(conjugated_mat, B, matC);
    destroy_csc(conjugated_mat);
    return status;
}
