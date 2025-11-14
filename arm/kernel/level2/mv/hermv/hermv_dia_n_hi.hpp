#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_dia.hpp"
#include "format/destroy_dia.hpp"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t
hermv_dia_n_hi(const TYPE alpha,
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
	}
	const ALPHA_INT diags = A->ndiag;
	for(ALPHA_INT i = 0; i < diags; ++i)
    {
		const ALPHA_INT dis = A->dis_data[i];
		if(dis == 0)
		{
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < m; ++j)
			{
				TYPE v;
				v = alpha_mul(alpha, ((TYPE *)A->val_data)[start + j]);
				y[j] = alpha_madde(y[j], v, x[j]);
			}
		}
		else if(dis > 0)
		{
			const ALPHA_INT row_start = 0;
			const ALPHA_INT col_start = dis;
			const ALPHA_INT nnz = m - dis;
			const ALPHA_INT start = i * A->lval;
			for(ALPHA_INT j = 0; j < nnz; ++j)
			{
				TYPE v,v_c;
				TYPE val_orig = ((TYPE *)A->val_data)[start + j];
				TYPE val_conj = cmp_conj(((TYPE *)A->val_data)[start + j]);
				v = alpha_mul(alpha, val_orig);
				v_c = alpha_mul(alpha, val_conj);
				y[row_start + j] = alpha_madde(y[row_start + j], v, x[col_start + j]);
				y[col_start + j] = alpha_madde(y[col_start + j], v_c, x[row_start + j]);
			}
		}
    }
    
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
