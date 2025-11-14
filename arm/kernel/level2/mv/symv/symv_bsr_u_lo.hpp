#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "stdio.h"
#include <stdlib.h>
#include <string.h>
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
symv_bsr_u_lo(const TYPE alpha,
			const internal_spmat A,
			const TYPE *x,
			const TYPE beta,
			TYPE *y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	TYPE *part = (TYPE*)malloc(A->rows * A->block_dim*sizeof(TYPE));
	memset(part, 0, A->rows * A->block_dim*sizeof(TYPE));
	for (ALPHA_INT j = 0; j < A->rows * A->block_dim; j++){
		part[j] = alpha_mul(y[j], beta);
		//part[j] = y[j]*beta;
		y[j] = alpha_setzero(y[j]);
	}
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		ALPHA_INT diag_block = 0;
    	for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_data[ai];
    	        if(col > i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first diag indx of the s-row in bolck[ai][col]
						// of A->value
						y[m_s+s/bs] = alpha_add(y[m_s+s/bs], x[s/bs+col*bs]);
						//y[m_s+s/bs] += x[s/bs+col*bs];
						for (ALPHA_INT s1 = s; s1 < s + s/bs; s1++){
							y[m_s+s/bs] = alpha_madde(y[m_s+s/bs], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[s1-s+col*bs]);
							//y[m_s+s/bs] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col*bs];
							y[s1-s+col*bs] = alpha_madde(y[s1-s+col*bs], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[m_s+s/bs]);
							//y[s1-s+col*bs] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s/bs];
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
							y[m_s+s/bs] = alpha_madde(y[m_s+s/bs], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[s1-s+col*bs]);
							//y[m_s+s/bs] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col*bs];
							y[s1-s+col*bs] = alpha_madde(y[s1-s+col*bs], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[m_s+s/bs]);
							//y[s1-s+col*bs] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s/bs];
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					y[m_s+st] = alpha_add(y[m_s+st], x[st+m_s]);
					//y[m_s+st] += x[st+m_s];
				}
			}
    	}
		for(ALPHA_INT k = 0; k < A->rows * A->block_dim; k++){
			part[k] = alpha_madd(y[k], alpha, part[k]);
			y[k] = part[k];
		}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
		ALPHA_INT diag_block = 0;
		for(ALPHA_INT i = 0; i < m_inner; ++i)
    	{
			ALPHA_INT m_s = i*bs;
    	    for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
    	    {
    	        const ALPHA_INT col = A->col_data[ai];
    	        if(col > i)
    	        {
    	            continue;
    	        }
    	        else if(col == i)
    	        {
					diag_block = 1;
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						y[m_s+s/bs] = alpha_madd(alpha, x[s/bs+col*bs], y[m_s+s/bs]);
						//y[m_s+s/bs] += alpha*x[s/bs+col*bs];
						for (ALPHA_INT s1 = s +s/bs +1; s1 < s + bs; s1++){
							// A->value[s1] is in [m_s+s1/bs][(i+ai)*bs+s/bs]
							y[m_s+s1-s] = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							y[m_s+s1-s] = alpha_mul(y[m_s+s1-s], x[col*bs+s/bs]);
							//y[m_s+s1-s] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col*bs+s/bs];
							y[col*bs+s/bs] = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							y[col*bs+s/bs] = alpha_mul(y[col*bs+s/bs], x[m_s+s1-s]);
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
							y[m_s+s1-s] = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							y[m_s+s1-s] = alpha_mul(y[m_s+s1-s], x[col*bs+s/bs]);
							//y[m_s+s1-s] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col*bs+s/bs];
							y[col*bs+s/bs] = alpha_mul(alpha, ((TYPE *)A->val_data)[s1+ai*bs*bs]);
							y[col*bs+s/bs] = alpha_mul(y[col*bs+s/bs], x[m_s+s1-s]);
							//y[col*bs+s/bs] += alpha*((TYPE *)A->val_data)[s1+ai*bs*bs]*x[m_s+s1-s];
						}
					}
    	        }
    	    }
            if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					y[m_s+st] = alpha_madd(alpha, x[st+m_s], y[m_s+st]);
					//y[m_s+st] += alpha*x[st+m_s];
				}
			}
    	}
    for(ALPHA_INT k = 0; k < A->rows * A->block_dim; k++){
			y[k] = alpha_add(y[k], part[k]);
			//y[k] = y[k] + part[k];
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
