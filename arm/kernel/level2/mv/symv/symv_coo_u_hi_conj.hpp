#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t
symv_coo_u_hi_conj(const TYPE alpha,
		              const internal_spmat A,
		              const TYPE *x,
		              const TYPE beta,
		              TYPE *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	const ALPHA_INT nnz = A->nnz;
	for(ALPHA_INT i = 0; i < m; ++i)
	{
		y[i] = alpha_mul(y[i], beta);
		y[i] = alpha_madde(y[i], alpha, x[i]);
	}
	for(ALPHA_INT i = 0; i < nnz; ++i)
	{
		const ALPHA_INT r = A->row_data[i];
		const ALPHA_INT c = A->col_data[i];
		if(r >= c)
		{
			continue;
		}
		TYPE v;
		v = cmp_conj(((TYPE *)A->val_data)[i]);
		v = alpha_mul(v, alpha);
		y[r] = alpha_madde(y[r], v, x[c]);
		y[c] = alpha_madde(y[c], v, x[r]);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
