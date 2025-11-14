#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_csc_n_hi_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_csc_n_lo(alpha, conjugated_mat, x, y);
    destroy_csc(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_n_lo_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_csc_n_hi(alpha, conjugated_mat, x, y);
    destroy_csc(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_u_hi_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_csc_u_lo(alpha, conjugated_mat, x, y);
    destroy_csc(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_u_lo_conj(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_csc<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_csc_u_hi(alpha, conjugated_mat, x, y);
    destroy_csc(conjugated_mat);
    return status;
}
