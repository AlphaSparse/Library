#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t
diagmv_dia_n(const TYPE alpha,
			const internal_spmat A,
			const TYPE *x,
			const TYPE beta,
			TYPE *y)
{
    const ALPHA_INT m = A->rows;
	const ALPHA_INT n = A->cols;
	if(m != n) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	const ALPHA_INT diags = A->ndiag;
	ALPHA_INT coll = -1;
	for(ALPHA_INT i = 0; i < diags; ++i)
	 {
		if(A->dis_data[i] == 0)
	 	{

			for(ALPHA_INT j = 0; j < m; ++j)
	 		{
				y[j] = alpha_mul(beta, y[j]);
				TYPE v;
				v = alpha_mul(alpha, ((TYPE *)A->val_data)[i * m + j]);
				y[j] = alpha_madd(v, x[j], y[j]);
				if( !(alpha_iszero(((TYPE *)A->val_data)[i * m + j])) ){
					coll = j + 1;
					break;
				}
			}
			for(ALPHA_INT j = coll ;j < m ; j++){
				TYPE val = ((TYPE *)A->val_data)[i * m + j];
				if(alpha_iszero(val)){
					continue;	
				}
				y[j] = alpha_mul(beta, y[j]);
				TYPE v;
				v = alpha_mul(alpha, val);
				y[j] = alpha_madd(v, x[j], y[j]);
			}
			break;
		}
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
