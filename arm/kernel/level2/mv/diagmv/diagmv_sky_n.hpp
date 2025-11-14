#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
diagmv_sky_n(const TYPE alpha,
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
		// y[i] *= beta;
	}
	for(ALPHA_INT i = 1; i < m + 1; ++i)
	{
		const ALPHA_INT indx = A->pointers[i] - 1;

		TYPE v;
		v = ((TYPE *)A->val_data)[indx];
		v = alpha_mul(v, x[i - 1]);
		y[i - 1] = alpha_madd(alpha, v, y[i - 1]);
		// y[i - 1] += alpha * v * x[i - 1];
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
