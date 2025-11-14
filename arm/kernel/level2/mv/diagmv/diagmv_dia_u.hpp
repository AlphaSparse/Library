#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t
diagmv_dia_u(const TYPE alpha,
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
		y[i] = alpha_madd(alpha, x[i], y[i]);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
