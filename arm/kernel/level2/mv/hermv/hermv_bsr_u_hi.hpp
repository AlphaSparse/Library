#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/destroy_bsr.hpp"
#include "alphasparse/util.h"
template <typename TYPE>
alphasparseStatus_t
hermv_bsr_u_hi(const TYPE alpha,
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
//	b_rows = ( b_rows*bs = = A->rows)?(b_rows):(b_rows+1);

	ALPHA_INT b_cols = A->cols;
//	b_cols = ( b_cols*bs = = A->cols)?(b_cols):(b_cols+1);
	
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
	TYPE temp_orig = TYPE{};
	TYPE temp_conj = TYPE{};
	
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
	{
		for(ALPHA_INT br = 0 ; br < b_rows; br++){
			row = br * bs;
			
			for(ALPHA_INT ai = A->row_data[br]; ai < A->row_data[br+1]; ++ai){
	            ALPHA_INT bc = A->col_data[ai];
				col = bc * bs;
				//block (br,bc)
				if(bc < br ){
					continue;
				}
				a0_idx = ai * bs2;
				// diagonal block containing diagonal entry
				if(bc == br){
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col) is unit
						//y[b_row + row] += alpha*((TYPE *)A->val_data)[a0_idx + (b_row + 1) * bs]*x[col + b_col];
						
						for(ALPHA_INT b_col = b_row + 1; b_col < bs; b_col++){
							
							val_orig = ((TYPE *)A->val_data)[a0_idx + b_row * bs + b_col];
							val_conj = cmp_conj(val_orig);
							temp_orig = alpha_mul(alpha, val_orig);
							temp_conj = alpha_mul(alpha, val_conj);

							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_conj , x[row + b_row]);

						}
					}
				}
				else{
					for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						for(ALPHA_INT b_col = 0; b_col < bs; b_col++){
							
							val_orig = ((TYPE *)A->val_data)[a0_idx + b_row * bs + b_col];
							val_conj = cmp_conj(val_orig);
							temp_orig = alpha_mul(alpha, val_orig);
							temp_conj = alpha_mul(alpha, val_conj);

							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_conj , x[row + b_row]);

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
			
			for(ALPHA_INT ai = A->row_data[br]; ai < A->row_data[br+1]; ++ai){
	            ALPHA_INT bc = A->col_data[ai];
				col = bc * bs;
				//block (br,bc)
				if(bc < br ){
					continue;
				}
				a0_idx = ai * bs2;
				// diagonal block containing diagonal entry
				if(bc == br){
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						//dignaol entry A(row+b_row,col+b_col) is unit
						//y[b_row + row] += alpha*((TYPE *)A->val_data)[a0_idx + (b_row + 1) * bs]*x[col + b_col];
						
						for(ALPHA_INT b_row = 0; b_row < b_col; b_row++){
							
							val_orig = ((TYPE *)A->val_data)[a0_idx + b_col * bs + b_row];
							val_conj = cmp_conj(val_orig);
							temp_orig = alpha_mul(alpha, val_orig);
							temp_conj = alpha_mul(alpha, val_conj);

							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_conj , x[row + b_row]);

						}
					}
				}
				else{
					for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
							
							val_orig = ((TYPE *)A->val_data)[a0_idx + b_col * bs + b_row];
							val_conj = cmp_conj(val_orig);
							temp_orig = alpha_mul(alpha, val_orig);
							temp_conj = alpha_mul(alpha, val_conj);

							y[b_row + row] = alpha_madde(y[b_row + row], temp_orig , x[col + b_col]);
							y[b_col + col] = alpha_madde(y[b_col + col], temp_conj , x[row + b_row]);

						}
					}
					
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
