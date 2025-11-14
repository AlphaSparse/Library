#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
diagmv_bsr_u(const TYPE alpha,
		             const internal_spmat A,
		             const TYPE *x,
		             const TYPE beta,
		             TYPE *y)
{
	const ALPHA_INT m = A->rows * A->block_dim;
	TYPE temp_1;
	temp_1 = alpha_setzero(temp_1);
	TYPE temp_2;
	temp_2 = alpha_setzero(temp_2);
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for(ALPHA_INT i = 0; i < m; ++i)
		{
			temp_1 = alpha_mul(alpha, x[i]);
			temp_2 = alpha_mul(beta, y[i]);
			y[i] = alpha_add(temp_1, temp_2);  
			//y[i] = beta * y[i] + alpha * x[i];
		} 
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for(ALPHA_INT i = 0; i < m; ++i)
		{
			temp_1 = alpha_mul(alpha, x[i]);
			temp_2 = alpha_mul(beta, y[i]);
			y[i] = alpha_add(temp_1, temp_2);
			//y[i] = beta * y[i] + alpha * x[i];
		} 
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
