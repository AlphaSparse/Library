#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
trmv_sky_n_lo_conj(const TYPE alpha,
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

    for(ALPHA_INT c = 0; c < n; ++c)
    {
        const ALPHA_INT col_start = A->pointers[c];
		const ALPHA_INT col_end = A->pointers[c + 1];
        ALPHA_INT col_indx = 1;

        for(ALPHA_INT ai = col_start; ai < col_end; ++ai)
        {
            ALPHA_INT col_eles = col_end - col_start;
            ALPHA_INT r = c - col_eles + col_indx;
            TYPE t;
            t = alpha_mul_3c(alpha, ((TYPE *)A->val_data)[ai]);
            y[r] = alpha_madde(y[r], t, x[c]);
			// y[r] += alpha* ((TYPE *)A->val_data)[ai] * x[c];
            col_indx ++;
        }
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
