#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
hermv_sky_n_hi_trans(const TYPE alpha,
	  const internal_spmat A,
	  const TYPE *x,
	  const TYPE beta,
	  TYPE *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	

	for(ALPHA_INT i = 0; i < m; ++i)
	{
		y[i] = alpha_mul(y[i], beta);
	}
	for(ALPHA_INT c = 0; c < n; ++c)
    {
		const ALPHA_INT col_start = A->pointers[c];
		const ALPHA_INT col_end = A->pointers[c + 1];
		ALPHA_INT col_indx = 1;
		for(ALPHA_INT i = col_start; i < col_end; i++)
		{
			ALPHA_INT col_eles = col_end - col_start;
			TYPE v;
			TYPE v_c;
			v_c = cmp_conj(((TYPE *)A->val_data)[i]);
			v_c = alpha_mul(v_c, alpha);
			v = ((TYPE *)A->val_data)[i];
			v = alpha_mul(v, alpha);
			if(i == col_end - 1)
			{
				ALPHA_INT r = c;
				y[r] = alpha_madde(y[r], v_c, x[c]);
				// y[r] += alpha * v * x[c];
			}
			else
			{
				ALPHA_INT r = c - col_eles + col_indx;
				y[r] = alpha_madde(y[r], v_c, x[c]);
				y[c] = alpha_madde(y[c], v, x[r]);
				// y[r] += alpha * v * x[c];
				// y[c] += alpha * v * x[r];
				col_indx ++;
			}
		}
    }
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
