#include "alphasparse/util.h"
#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t gemv_dia(const TYPE alpha,
		         const internal_spmat A,
		         const TYPE* x,
		         const TYPE beta,
		         TYPE* y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	for (ALPHA_INT i = 0; i < m; ++i)
	{
		y[i] = alpha_mul(y[i], beta);
	}
	const ALPHA_INT diags = A->ndiag;
    for (ALPHA_INT i = 0; i < diags; i++)
    {
        const ALPHA_INT dis = A->dis_data[i];
		const ALPHA_INT row_start = dis>0?0:-dis;
		const ALPHA_INT col_start = dis>0?dis:0;
		const ALPHA_INT nnz = (m - row_start)<(n - col_start)?(m - row_start):(n - col_start);
		const ALPHA_INT start = i * A->lval;
		for(ALPHA_INT j = 0; j < nnz; ++j)
		{
			TYPE v;
			v = alpha_mul(((TYPE *)A->val_data)[start + row_start + j], alpha);
			y[row_start + j] = alpha_madd(v, x[col_start + j], y[row_start + j]);
		}
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_dia_conj(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE* x,
		               const TYPE beta,
		               TYPE* y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	for (ALPHA_INT i = 0; i < n; ++i)
	{
		y[i] = alpha_mul(y[i], beta);
	}
	const ALPHA_INT diags = A->ndiag;
    for (ALPHA_INT i = 0; i < diags; i++)
    {
        const ALPHA_INT dis = A->dis_data[i];
		const ALPHA_INT row_start = dis>0?0:-dis;
		const ALPHA_INT col_start = dis>0?dis:0;
		const ALPHA_INT nnz = (m - row_start)<(n - col_start)?(m - row_start):(n - col_start);
		const ALPHA_INT start = i * A->lval;
		for(ALPHA_INT j = 0; j < nnz; ++j)
		{
			TYPE v;
			v = alpha_mul_3c(alpha, ((TYPE *)A->val_data)[start + row_start + j]);
			y[col_start + j] = alpha_madd(v, x[row_start + j], y[col_start + j]);
		}
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_dia_trans(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE* x,
		               const TYPE beta,
		               TYPE* y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	for (ALPHA_INT i = 0; i < n; ++i)
	{
		y[i] = alpha_mul(y[i], beta);
	}
	const ALPHA_INT diags = A->ndiag;
    for (ALPHA_INT i = 0; i < diags; i++)
    {
        const ALPHA_INT dis = A->dis_data[i];
		const ALPHA_INT row_start = dis>0?0:-dis;
		const ALPHA_INT col_start = dis>0?dis:0;
		const ALPHA_INT nnz = (m - row_start)<(n - col_start)?(m - row_start):(n - col_start);
		const ALPHA_INT start = i * A->lval;
		for(ALPHA_INT j = 0; j < nnz; ++j)
		{
			TYPE v;
			v = alpha_mul(alpha, ((TYPE *)A->val_data)[start + row_start + j]);
			y[col_start + j] = alpha_madd(v, x[row_start + j], y[col_start + j]);
		}
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}