#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

#include "format/transpose_dia.hpp"
#include "format/transpose_conj_dia.hpp"
#include "format/destroy_dia.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_dia_n_hi_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_dia<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_dia_n_lo_plain(alpha, transposed_mat, x, y);
    destroy_dia(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_n_lo_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_dia<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_dia_n_hi_plain(alpha, transposed_mat, x, y);
    destroy_dia(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_u_hi_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_dia<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_dia_u_lo_plain(alpha, transposed_mat, x, y);
    destroy_dia(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_u_lo_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_dia<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_dia_u_hi_plain(alpha, transposed_mat, x, y);
    destroy_dia(transposed_mat);
    return status;
}