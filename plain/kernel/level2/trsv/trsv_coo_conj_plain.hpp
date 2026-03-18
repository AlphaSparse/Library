#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_coo_n_hi_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_coo<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_coo_n_lo_plain(alpha, conjugated_mat->rows, conjugated_mat->cols, conjugated_mat->nnz, conjugated_mat->row_data, conjugated_mat->col_data, (TYPE*)conjugated_mat->val_data, x, y);
    destroy_coo(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_coo_n_lo_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_coo<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_coo_n_hi_plain(alpha, conjugated_mat->rows, conjugated_mat->cols, conjugated_mat->nnz, conjugated_mat->row_data, conjugated_mat->col_data, (TYPE*)conjugated_mat->val_data, x, y);
    destroy_coo(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_coo_u_hi_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_coo<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_coo_u_lo_plain(alpha, conjugated_mat->rows, conjugated_mat->cols, conjugated_mat->nnz, conjugated_mat->row_data, conjugated_mat->col_data, (TYPE*)conjugated_mat->val_data, x, y);
    destroy_coo(conjugated_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_coo_u_lo_conj_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat conjugated_mat;
    transpose_conj_coo<TYPE>(A, &conjugated_mat);
    alphasparseStatus_t status = trsv_coo_u_hi_plain(alpha, conjugated_mat->rows, conjugated_mat->cols, conjugated_mat->nnz, conjugated_mat->row_data, conjugated_mat->col_data, (TYPE*)conjugated_mat->val_data, x, y);
    destroy_coo(conjugated_mat);
    return status;
}
