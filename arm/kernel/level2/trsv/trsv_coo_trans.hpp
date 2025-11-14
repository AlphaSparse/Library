#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

template <typename TYPE>
alphasparseStatus_t trsv_coo_u_lo_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_coo<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_coo_u_hi(alpha, transposed_mat->rows, transposed_mat->cols, transposed_mat->nnz, transposed_mat->row_data, transposed_mat->col_data, (TYPE*)transposed_mat->val_data, x, y);
    destroy_coo(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_coo_u_hi_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_coo<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_coo_u_lo(alpha, transposed_mat->rows, transposed_mat->cols, transposed_mat->nnz, transposed_mat->row_data, transposed_mat->col_data, (TYPE*)transposed_mat->val_data, x, y);
    destroy_coo(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_coo_n_hi_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_coo<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_coo_n_lo(alpha, transposed_mat->rows, transposed_mat->cols, transposed_mat->nnz, transposed_mat->row_data, transposed_mat->col_data, (TYPE*)transposed_mat->val_data, x, y);
    destroy_coo(transposed_mat);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_coo_n_lo_trans(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    internal_spmat transposed_mat;
    transpose_coo<TYPE>(A, &transposed_mat);
    alphasparseStatus_t status = trsv_coo_n_hi(alpha, transposed_mat->rows, transposed_mat->cols, transposed_mat->nnz, transposed_mat->row_data, transposed_mat->col_data, (TYPE*)transposed_mat->val_data, x, y);
    destroy_coo(transposed_mat);
    return status;
}