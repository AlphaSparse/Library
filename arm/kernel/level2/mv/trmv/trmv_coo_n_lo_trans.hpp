#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t
trmv_coo_n_lo_trans(const TYPE alpha,
                            const internal_spmat A,
                            const TYPE *x,
                            const TYPE beta,
                            TYPE *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	const ALPHA_INT nnz = A->nnz;
	for(int i = 0; i < m; ++i)
	{ 
		y[i] = alpha_mul(y[i], beta);
	}
    for(ALPHA_INT i = 0; i < nnz; ++i)
    { 
		const ALPHA_INT r = A->row_data[i];
		const ALPHA_INT c = A->col_data[i];
		if(r >= c)
	 	{
			TYPE v;
			v = alpha_mul(((TYPE *)A->val_data)[i], x[r]);
			y[c] = alpha_madde(y[c], alpha, v);
		}
    }

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
