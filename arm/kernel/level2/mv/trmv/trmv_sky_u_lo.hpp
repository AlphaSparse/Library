#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
trmv_sky_u_lo(const TYPE alpha,
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
    for(ALPHA_INT r = 0; r < m; ++r)
    {
		const ALPHA_INT row_start = A->pointers[r];
		const ALPHA_INT row_end = A->pointers[r + 1];
		ALPHA_INT row_indx = 1;
		for(ALPHA_INT i = row_start; i < row_end; i++)
		{
			ALPHA_INT row_eles = row_end - row_start;
            ALPHA_INT c = r - row_eles + row_indx;
            if(i == row_end - 1)
			{
				y[r] = alpha_madde(y[r], alpha, x[c]);
                // y[r] += alpha * x[c];
			}
            else
			{
				TYPE t;
				t = alpha_mul(alpha, ((TYPE *)A->val_data)[i]);
				y[r] = alpha_madde(y[r], t, x[c]);
				// const TYPE v = ((TYPE *)A->val_data)[i];
				// y[r] += alpha * v * x[c];
			}
            row_indx ++;
		}
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
