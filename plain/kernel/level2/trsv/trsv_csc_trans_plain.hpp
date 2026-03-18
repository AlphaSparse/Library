#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_csc_n_hi_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat trans_mat;
    transpose_csc<TYPE>(A, &trans_mat);
    alphasparseStatus_t status = trsv_csc_n_lo_plain(alpha, trans_mat, x, y);
    destroy_csc(trans_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_n_lo_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat trans_mat;
    transpose_csc<TYPE>(A, &trans_mat);
    alphasparseStatus_t status = trsv_csc_n_hi_plain(alpha, trans_mat, x, y);
    destroy_csc(trans_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_u_hi_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat trans_mat;
    transpose_csc<TYPE>(A, &trans_mat);
    alphasparseStatus_t status = trsv_csc_u_lo_plain(alpha, trans_mat, x, y);
    destroy_csc(trans_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_u_lo_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat trans_mat;
    transpose_csc<TYPE>(A, &trans_mat);
    alphasparseStatus_t status = trsv_csc_u_hi_plain(alpha, trans_mat, x, y);
    destroy_csc(trans_mat);
    return status;
}
