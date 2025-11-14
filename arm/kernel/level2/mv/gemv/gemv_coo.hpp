#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t gemv_coo(const TYPE alpha,
		         const internal_spmat A,
		         const TYPE* x,
		         const TYPE beta,
		         TYPE* y)
{
    ALPHA_INT m = A->rows;
	ALPHA_INT nnz = A->nnz;
	for (ALPHA_INT i = 0; i < m; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        ALPHA_INT r = A->row_data[i];
		ALPHA_INT c = A->col_data[i];
		TYPE v;
		v = alpha_mul(((TYPE *)A->val_data)[i], x[c]);
		y[r] = alpha_madd(alpha, v, y[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_coo_conj(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE* x,
		               const TYPE beta,
		               TYPE* y)
{
    ALPHA_INT m = A->cols;
	ALPHA_INT nnz = A->nnz;
	for (ALPHA_INT i = 0; i < m; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        ALPHA_INT r = A->row_data[i];
		ALPHA_INT c = A->col_data[i];
		TYPE v;
		v = cmp_conj(((TYPE *)A->val_data)[i]);
		v = alpha_mul(v, x[r]);
		y[c] = alpha_madd(alpha, v, y[c]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_coo_trans(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE* x,
		               const TYPE beta,
		               TYPE* y)
{
    ALPHA_INT m = A->cols;
	ALPHA_INT nnz = A->nnz;
	for (ALPHA_INT i = 0; i < m; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
        ALPHA_INT r = A->row_data[i];
		ALPHA_INT c = A->col_data[i];
		TYPE v;
		v = alpha_mul(((TYPE *)A->val_data)[i], x[r]);
		y[c] = alpha_madd(alpha, v, y[c]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}