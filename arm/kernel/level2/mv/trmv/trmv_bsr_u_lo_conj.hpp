#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
trmv_bsr_u_lo_conj(const TYPE alpha,
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
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		ALPHA_INT diag_block = 0;
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++){	
				ALPHA_INT row = A->col_data[ai];
				ALPHA_INT m_s = row*bs;
				if (row > i){
					continue;
				}else if (row == i){
					diag_block = 1;
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						y[m_s+s/bs] = alpha_madde(y[m_s+s/bs], alpha, x[col+s/bs]);
						//y[m_s+s/bs] += alpha*x[col+s/bs];
						for (int s1 = s; s1 < s +s/bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][(i+ai)*bs+s/bs]
							tmp = alpha_conj(((TYPE *)A->val_data)[s1+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[col+s/bs]);
							y[m_s+s1-s] = alpha_add(y[m_s+s1-s], tmp);
							//y[m_s+s1-s] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col+s/bs];
						}
					}
				}else {
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (int s1 = s; s1 < s+bs; s1++){
							tmp = alpha_conj(((TYPE *)A->val_data)[s1+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[col+s/bs]);
							y[m_s+s1-s] = alpha_add(y[m_s+s1-s], tmp);
							//y[m_s+s1-s] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col+s/bs];
						}
					}
				}if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						y[m_s+s] = alpha_madde(y[m_s+s], alpha, x[m_s+s]);
						//y[m_s+s] += alpha*x[m_s+s];
				}
			}
		}
	}else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        ALPHA_INT diag_block = 0;
		for (ALPHA_INT i = 0; i < m_inner; i++){
			ALPHA_INT col = i*bs;
			for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++){	
				ALPHA_INT row = A->col_data[ai];
				ALPHA_INT m_s = row*bs;
				if (row > i ){
					continue;
				}else if (row == i){
                    diag_block = 1;
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						y[m_s+s/bs] = alpha_madde(y[m_s+s/bs], alpha, x[s/bs+col]);
						//y[m_s+s/bs] += alpha*x[s/bs+col];	
						for (int s1 = s + 1 + s/bs; s1 < s+bs; s1++){
							tmp = alpha_conj(((TYPE *)A->val_data)[s1+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[s1-s+col]);
							y[m_s+s/bs] = alpha_add(y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col];
						}
					}
				}else {
					for (int s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (int s1 = s; s1 < s+bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][col*bs+s1-ai*bs*bs-s]
							tmp = alpha_conj(((TYPE *)A->val_data)[s1+ai*bs*bs]);
							tmp = alpha_mul(alpha, tmp);
							tmp = alpha_mul(tmp, x[s1-s+col]);
							y[m_s+s/bs] = alpha_add(y[m_s+s/bs], tmp);
							//y[m_s+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col];
						}
					}
				}
                if (diag_block == 0){
					for (ALPHA_INT s = 0; s < bs; s++)
						y[m_s+s] = alpha_madde(y[m_s+s], alpha, x[m_s+s]);
						//y[m_s+s] += alpha*x[m_s+s];
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
