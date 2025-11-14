#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <stdio.h>

template <typename TYPE>
alphasparseStatus_t
diagmv_bsr_n(const TYPE alpha,
		             const internal_spmat A,
		             const TYPE *x,
		             const TYPE beta,
		             TYPE *y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;
	if (m_inner != n_inner) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	for (ALPHA_INT j = 0; j < A->rows * A->block_dim; j++){
		y[j] = alpha_mul(y[j], beta); 
		//y[j] *= beta;
	}
	// For matC, block_layout is defaulted as row_major
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
        ALPHA_INT not_hit_hp = 1;
		for(ALPHA_INT i = 0; i < m_inner; i++){
		ALPHA_INT diag_block = 0;
		TYPE temp;
		temp = alpha_setzero(temp);
			for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1];ai++){
				// the block is the diag one
				if(A->col_data[ai] == i){
					diag_block = 1;
                    not_hit_hp = 0;
					for(ALPHA_INT bi = 0; bi < bs; bi++){
						temp = alpha_mul(x[i*bs+bi], ((TYPE *)A->val_data)[ai*bs*bs+(bs+1)*bi]);
						y[i*bs+bi] = alpha_madd(alpha, temp, y[i*bs+bi]); 
						//y[i*bs+bi] += alpha*x[i*bs+bi]*((TYPE *)A->val_data)[ai*bs*bs+(bs+1)*bi];
					}
				}
			}if (diag_block == 0 && not_hit_hp == 0){
				for (ALPHA_INT s = 0; s < bs; s++){
					y[i*bs+s] = x[i*bs+s];
				}
			}
		}
	}
	// For Fortran, block_layout is defaulted as col_major
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        ALPHA_INT not_hit_hp = 1;
		for(ALPHA_INT i = 0; i < m_inner; i++){
		ALPHA_INT diag_block = 0;
		TYPE temp;
		temp = alpha_setzero(temp);
			for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1];ai++){
				// the block is the diag one
				if(A->col_data[ai] == i){
					diag_block = 1;
                    not_hit_hp = 0;
					for(ALPHA_INT bi = 0; bi < bs; bi++){
						temp = alpha_mul(x[i*bs+bi], ((TYPE *)A->val_data)[ai*bs*bs+(bs+1)*bi]);
						y[i*bs+bi] = alpha_madd(alpha, temp, y[i*bs+bi]); 
						//y[i*bs+bi] += alpha*x[i*bs+bi]*((TYPE *)A->val_data)[ai*bs*bs+(bs+1)*bi];
					}
				}
			}if (diag_block == 0 && not_hit_hp == 0){
				for (ALPHA_INT s = 0; s < bs; s++){
					y[i*bs+s] = x[i*bs+s];
				}
			}
		}
	}else return ALPHA_SPARSE_STATUS_INVALID_VALUE;
	return ALPHA_SPARSE_STATUS_SUCCESS;
 }
