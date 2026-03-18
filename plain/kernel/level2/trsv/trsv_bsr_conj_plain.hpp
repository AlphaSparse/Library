#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_bsr_n_hi_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_bsr_n_lo_plain(alpha, conjugated_mat, x, y);
    destroy_bsr(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_n_lo_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_bsr_n_hi_plain(alpha, conjugated_mat, x, y);
    destroy_bsr(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_u_hi_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_bsr_u_lo_plain(alpha, conjugated_mat, x, y);
    destroy_bsr(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_u_lo_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_bsr_u_hi_plain(alpha, conjugated_mat, x, y);
    destroy_bsr(conjugated_mat);
    return status;
}