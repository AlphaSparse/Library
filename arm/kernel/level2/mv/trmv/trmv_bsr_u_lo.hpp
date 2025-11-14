#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "stdio.h"
#include <stdlib.h>
#include "alphasparse/util.h"

#ifdef BSR_USE_FAKE
template <typename TYPE>
alphasparseStatus_t
trmv_bsr_u_lo(const TYPE alpha,
		              const internal_spmat A,
		              const TYPE *x,
		              const TYPE beta,
		              TYPE *y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
    if(m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	TYPE *part = (TYPE*)malloc(A->rows * A->block_dim * sizeof(TYPE));
	for (ALPHA_INT j = 0; j < A->rows * A->block_dim; j++){
		alpha_mul(part[j], y[j], beta);
		//part[j] = y[j]*beta;
		alpha_setzero(y[j]);
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
						alpha_add(y[m_s+s/bs], y[m_s+s/bs], x[s/bs+col*bs]);
						//y[m_s+s/bs] += x[s/bs+col*bs];
						for (ALPHA_INT s1 = s; s1 < s + s/bs; s1++){
							alpha_madde(y[m_s+s/bs], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[s1-s+col*bs]);
							//y[m_s+s/bs] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col*bs];
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
							alpha_madde(y[m_s+s/bs], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[s1-s+col*bs]);
							//y[m_s+s/bs] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[s1-s+col*bs];
						}
					}
    	        }
    	    }if (diag_block == 0){
				for (ALPHA_INT st = 0; st < bs; st++){
					alpha_add(y[m_s+st], y[m_s+st], x[st+m_s]);
					//y[m_s+st] += x[st+m_s];
				}
			}
    	}
		for(ALPHA_INT k = 0; k < A->rows * A->block_dim; k++){
			alpha_madde(part[k], y[k], alpha);
			//y[k].imag = part[k].imag;
			y[k] = part[k];
			//y[k] = y[k]*alpha + part[k];
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
						alpha_add(y[m_s+s/bs], y[m_s+s/bs], x[s/bs+col*bs]);
						//y[m_s+s/bs] += x[s/bs+col*bs];
						for (ALPHA_INT s1 = s +s/bs +1; s1 < s + bs; s1++){
							alpha_madde(y[m_s+s1-s], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[col*bs+s/bs]);
							//y[m_s+s1-s] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col*bs+s/bs];
						}
					}
    	        }
    	        else
    	        {
					for (ALPHA_INT s = 0; s < bs*bs; s=s+bs){
						// 's1' is the first indx of the s-row in bolck[ai][col]
						// of A->value
						for (ALPHA_INT s1 = s; s1 < s+bs; s1++){
							alpha_madde(y[m_s+s1-s], ((TYPE *)A->val_data)[s1+ai*bs*bs], x[col*bs+s/bs]);
							//y[m_s+s1-s] += ((TYPE *)A->val_data)[s1+ai*bs*bs]*x[col*bs+s/bs];
						}
					}
    	        }
    	    }
            if (diag_block == 0){
					for (ALPHA_INT st = 0; st < bs; st++){
						alpha_add(y[m_s+st], y[m_s+st], x[st+m_s]);
						//y[m_s+st] += x[st+m_s];
					}
			}
    	}
		for(ALPHA_INT k = 0; k < A->rows * A->block_dim; k++){
			alpha_madde(part[k], alpha, y[k]);
			y[k].imag = part[k].imag;
			//y[k] = y[k]*alpha + part[k];
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}

#else
template <typename TYPE>
alphasparseStatus_t
trmv_bsr_u_lo(const TYPE alpha,
		              const internal_spmat A,
		              const TYPE *x,
		              const TYPE beta,
		              TYPE *y)
{
	const ALPHA_INT m = A->rows * A->block_dim;
	const ALPHA_INT n = A->cols * A->block_dim;
	const ALPHA_INT bs = A -> block_dim;
	const ALPHA_INT bs2=bs * bs;
    // assert(m==n);
	ALPHA_INT b_rows = A->rows ;
	ALPHA_INT b_cols = A->cols;
    if(b_rows != b_cols) return ALPHA_SPARSE_STATUS_INVALID_VALUE;

	for (ALPHA_INT j = 0; j < A->rows * A->block_dim; j++){
		y[j] = alpha_mul(y[j], beta);
		y[j] = alpha_madde(y[j], alpha, x[j]);
		//y[j] *= beta;
	}
	ALPHA_INT a0_idx = -1;
	ALPHA_INT row = -1;
	ALPHA_INT col = -1;
	TYPE val_orig ,val_conj;
	TYPE temp_orig;
	temp_orig = alpha_setzero(temp_orig);

	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		for(ALPHA_INT br = 0 ; br < b_rows; br++){
			row = br * bs;
			ALPHA_INT block_start = A->row_data[br],block_end = A->row_data[br+1];
			ALPHA_INT lower_end = alpha_upper_bound(&A->col_data[block_start],&A->col_data[block_end],br)-A->col_data;
			for(ALPHA_INT ai = block_start; ai < lower_end;ai++){
				ALPHA_INT bc = A->col_data[ai];
				col = bc * bs;
				a0_idx = ai * bs2;
				// diagonal block containing diagonal entry
				if(bc == br){
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						for(ALPHA_INT b_col = 0; b_col < b_row; b_col++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_row * bs + b_col]);
							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
						}
					}
				}
				else{
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						ALPHA_INT b_col = 0;
						for( ; b_col < bs ; b_col++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_row * bs + b_col]);
							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
						}
					}
				}
			}
		}
	}
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
	{
    	for(ALPHA_INT br = 0 ; br < b_rows; br++){
			row = br * bs;
			ALPHA_INT block_start = A->row_data[br],block_end = A->row_data[br+1];
			ALPHA_INT lower_end = alpha_upper_bound(&A->col_data[block_start],&A->col_data[block_end],br)-A->col_data;

			for(ALPHA_INT ai = block_start; ai < lower_end; ++ai){
	            ALPHA_INT bc = A->col_data[ai];
				col = bc * bs;
				a0_idx = ai * bs2;
				// diagonal block containing diagonal entry
				if(bc == br){
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						for(ALPHA_INT b_row = b_col; b_row < bs; b_row++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_col * bs + b_row]);
							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
						}
					}
				}
				else{
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_col * bs + b_row]);
							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
						}
					}
					
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
#endif
