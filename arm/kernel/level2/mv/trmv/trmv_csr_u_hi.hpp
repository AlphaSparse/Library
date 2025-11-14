#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t
trmv_csr_u_hi(const TYPE alpha,
	  const internal_spmat A,
	  const TYPE *x,
	  const TYPE beta,
	  TYPE *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;

	for(ALPHA_INT i = 0;i < m; ++i)
	{
		TYPE tmp = x[i];
		for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
		{
			const ALPHA_INT col = A->col_data[ai];
			if(col <= i)
			{
				continue;
			}
			else
			{
				tmp = alpha_madd(((TYPE *)A->val_data)[ai], x[col], tmp);
			}
		}
		tmp = alpha_mul(tmp, alpha);
        y[i] = alpha_mul(y[i], beta);
        y[i] = alpha_add(y[i], tmp);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
