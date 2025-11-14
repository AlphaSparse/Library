#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
symv_sky_n_lo(const TYPE alpha,
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
		y[i] = alpha_mul(y[i], beta);
	}
    for(ALPHA_INT r = 0; r < m; ++r)
    {
		const ALPHA_INT row_start = A->pointers[r];
		const ALPHA_INT row_end = A->pointers[r + 1];
		ALPHA_INT row_indx = 1;
		for(ALPHA_INT i = row_start; i < row_end; i++)
		{
			ALPHA_INT row_eles = row_end - row_start;
			TYPE v;
			v = ((TYPE *)A->val_data)[i];
			v = alpha_mul(v, alpha);
			if(i == row_end - 1)
			{
				ALPHA_INT c = r;
				y[r] = alpha_madde(y[r], v, x[c]);
				// y[r] += alpha * v * x[c];
			}
			else
			{
				ALPHA_INT c = r - row_eles + row_indx;
				y[r] = alpha_madde(y[r], v, x[c]);
				y[c] = alpha_madde(y[c], v, x[r]);
				// y[r] += alpha * v * x[c];
				// y[c] += alpha * v * x[r];
				row_indx ++;
			}
		}
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
