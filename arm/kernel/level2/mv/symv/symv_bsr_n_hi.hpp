#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "stdio.h"
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
symv_bsr_n_hi(const TYPE alpha,
			const internal_spmat A,
			const TYPE *x,
			const TYPE beta,
			TYPE *y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	TYPE temp;
	temp = alpha_setzero(temp);
	for (ALPHA_INT j = 0; j < A->rows * A->block_dim; j++){
		y[j] = alpha_mul(y[j], beta);
		//y[j] *= beta;
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
    	for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_data[ai];
    	        if(col < i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s + s/bs; s1 < s+bs; s1++){
							temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							temp = alpha_mul(temp, x[s1-s+col*bs]);
							y[m_s+s/bs] = alpha_add(y[m_s+s/bs], temp);
							//y[m_s+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col*bs];
							if(s1 != s+s/bs) {
								temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
								temp = alpha_mul(temp, x[m_s+s/bs]);
								y[s1-s+col*bs] = alpha_add(y[s1-s+col*bs], temp);
							}
							//y[s1-s+col*bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s/bs];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][col*bs+s1-ai*bs*bs-s]
							temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							temp = alpha_mul(temp, x[s1-s+col*bs]);
							y[m_s+s/bs] = alpha_add(y[m_s+s/bs], temp);
							//y[m_s+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col*bs];
							temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							temp = alpha_mul(temp, x[m_s+s/bs]);
							y[s1-s+col*bs] = alpha_add(y[s1-s+col*bs], temp);
							//y[s1-s+col*bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s/bs];
						}
					}
    	        }
    	    }
    	}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
		for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_data[ai];
    	        if(col < i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 <= s +s/bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][(i+ai)*bs+s/bs]
							temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							temp = alpha_mul(temp, x[col*bs+s/bs]);
							y[m_s+s1-s] = alpha_add(y[m_s+s1-s], temp);
							//y[m_s+s1-s] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col*bs+s/bs];
							if(s1 != s+s/bs) {
								temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
								temp = alpha_mul(temp, x[m_s+s1-s]);
								y[col*bs+s/bs] = alpha_add(y[col*bs+s/bs], temp);
							}
							//y[col*bs+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s1-s];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							temp = alpha_mul(temp, x[col*bs+s/bs]);
							y[m_s+s1-s] = alpha_add(y[m_s+s1-s], temp);
							//y[m_s+s1-s] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col*bs+s/bs];
							temp = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							temp = alpha_mul(temp, x[m_s+s1-s]);
							y[col*bs+s/bs] = alpha_add(y[col*bs+s/bs], temp);
							//y[col*bs+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s1-s];
						}
					}
    	        }
    	    }
    	}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
