#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
trmv_bsr_n_hi_conj(const TYPE alpha,
				const internal_spmat A,
				const TYPE *x,
				const TYPE beta,
				TYPE *y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	TYPE tmp;
	tmp = alpha_setzero(tmp);
	for (ALPHA_INT j = 0; j < A->rows * A->block_dim; j++){
		y[j] = alpha_mul(y[j], beta);
		//y[j] *= beta;
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++){	
				ALPHA_INT row = A->col_data[ai];
				ALPHA_INT m_s = row*bs;
				if (row < i){
					continue;
				}else if (row == i){
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s + s/bs; st < s+bs; st++){
							tmp = alpha_conj(((TYPE *)A->val_data)[st+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[col+s/bs]);
							y[m_s+st-s] = alpha_add(y[m_s+st-s], tmp);
							//y[m_s+st-s] += alpha*((TYPE *)A->val_data)[st+ai*bs*bs]*x[col+s/bs];
						}
					}
				}else{
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st < s+bs; st++){
							tmp = alpha_conj(((TYPE *)A->val_data)[st+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[col+s/bs]);
							y[m_s+st-s] = alpha_add(y[m_s+st-s], tmp);
							//y[m_s+st-s] += alpha*((TYPE *)A->val_data)[st+ai*bs*bs]*x[col+s/bs];
						}
					}
				}
			}
		}
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++){
				ALPHA_INT row = A->col_data[ai];
				ALPHA_INT m_s = row*bs;
				if (row < i){
					continue;
				}else if (row == i){
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st <= s+s/bs; st++){
							tmp = alpha_conj(((TYPE *)A->val_data)[st+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[col+st-s]);
							y[m_s+s/bs] = alpha_add(y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*((TYPE *)A->val_data)[st+ai*bs*bs]*x[col+st-s];
						}
					}
				}else{
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						for(ALPHA_INT st = s; st < s+bs; st++){
							tmp = alpha_conj(((TYPE *)A->val_data)[st+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[col+st-s]);
							y[m_s+s/bs] = alpha_add(y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*((TYPE *)A->val_data)[st+ai*bs*bs]*x[col+st-s];
						}
					}
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
