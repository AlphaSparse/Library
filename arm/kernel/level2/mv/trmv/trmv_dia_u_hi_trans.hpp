#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t
trmv_dia_u_hi_trans(const TYPE alpha,
				const internal_spmat A,
				const TYPE *x,
				const TYPE beta,
				TYPE *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		y[i] = alpha_mul(beta, y[i]);
		y[i] = alpha_madde(y[i], alpha, x[i]);
	}
	const ALPHA_INT diags = A->ndiag;
	for(ALPHA_INT i = 0; i < diags; ++i)
	{
		const ALPHA_INT dis = A->dis_data[i];
		if(dis > 0)
		{
			const ALPHA_INT row_start = 0;
			const ALPHA_INT col_start = dis;
			const ALPHA_INT nnz = m - dis;
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < nnz; ++j)
			{
				TYPE v;
				v= alpha_mul(alpha, ((TYPE *)A->val_data)[start + j]);
				y[col_start + j] = alpha_madde(y[col_start + j], v, x[row_start + j]);
		 	}
		} 
	}

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
