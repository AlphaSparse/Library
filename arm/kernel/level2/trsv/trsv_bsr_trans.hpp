#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_bsr_u_lo_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_bsr<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_bsr_u_hi(alpha, transposed_mat, x, y);
    destroy_bsr(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_n_hi_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_bsr<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_bsr_n_lo(alpha, transposed_mat, x, y);
    destroy_bsr(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_n_lo_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_bsr<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_bsr_n_hi(alpha, transposed_mat, x, y);
    destroy_bsr(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_u_hi_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_bsr<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_bsr_u_lo(alpha, transposed_mat, x, y);
    destroy_bsr(transposed_mat);
    return status;
}