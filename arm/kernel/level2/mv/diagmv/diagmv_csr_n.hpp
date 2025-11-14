#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t
diagmv_csr_n(const TYPE alpha,
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
		for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
		{
			const ALPHA_INT col = A->col_data[ai];
			if(i == col)
			{
				TYPE tmp;
        		tmp = alpha_setzero(tmp);
				tmp = alpha_mul(alpha, ((TYPE *)A->val_data)[ai]);
				y[i] = alpha_madd(tmp, x[col], y[i]);
				// y[i] += alpha * ((TYPE *)A->val_data)[ai] * x[col];
				break;
			}
		}
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
