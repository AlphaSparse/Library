#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
trmv_bsr_n_lo_trans(const TYPE alpha,
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
						temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_row  * ( bs+1 ) ]);
						y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_row]);
						for(ALPHA_INT b_col = 0; b_col < b_row; b_col++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_row * bs + b_col]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);
						}
					}
				}
				else{
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						ALPHA_INT b_col = 0;
						for( ; b_col < bs ; b_col++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_row * bs + b_col]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);
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
						temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_col  * ( bs+1 ) ]);
						y[b_col + row] = alpha_madde(y[b_col + row], temp_orig , x[b_col + col]);
						for(ALPHA_INT b_row = b_col; b_row < bs; b_row++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_col * bs + b_row]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);
						}
					}
				}
				else{
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
							temp_orig = alpha_mul(alpha, ((TYPE *)A->val_data)[a0_idx + b_col * bs + b_row]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_orig , x[row + b_row]);
						}
					}
					
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}