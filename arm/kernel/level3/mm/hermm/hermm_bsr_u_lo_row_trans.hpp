#include"alphasparse/kernel.h"
#include"alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_bsr_u_lo_row_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows * mat->block_dim; i++)
        for(ALPHA_INT j = 0; j < columns; j++){
            y[j + i * ldy] = alpha_mul(y[j + i * ldy], beta);
            y[j + i * ldy] = alpha_madde(y[j + i * ldy], alpha, x[j + i * ldy]);
        }
            
    const ALPHA_INT m = mat->rows * mat->block_dim;
    const ALPHA_INT n = mat->cols * mat->block_dim;

    const ALPHA_INT bs = mat->block_dim;
    const ALPHA_INT bs2 = bs * bs;

    ALPHA_INT a0_idx = -1;
	ALPHA_INT col = -1;
	J val_orig ,val_conj;
	J temp_orig = J{};
	J temp_conj = J{};
    
    if(mat->block_layout== ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
//        printf("ALPHA_SPARSE_LAYOUT_ROW_MAJOR \n");
        for(ALPHA_INT row = 0 ; row < m ; row += bs){
            ALPHA_INT br = row / bs;
            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai){
                const ALPHA_INT bc = mat->col_data[ai];
                const ALPHA_INT col = bc * bs;
                if(bc > br ){
					continue;
				}
                a0_idx = ai * bs2;
                if(bc == br){
                    for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_col = 0; b_col < b_row; b_col++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = ((J *)mat->val_data)[a0_idx + b_row * bs + b_col];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);

                                y[index2(b_row + row,c,ldy)] = alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                y[index2(b_col + col,c,ldy)] = alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                        
                    }
                }
                else{
                    for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_col = 0; b_col < bs; b_col++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = ((J *)mat->val_data)[a0_idx + b_row * bs + b_col];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);

                                y[index2(b_row + row,c,ldy)] = alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                y[index2(b_col + col,c,ldy)] = alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                    }
                }
            }
        }
    }
    
    else if( mat->block_layout== ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        for(ALPHA_INT row = 0 ; row < m ; row += bs){
            ALPHA_INT br = row / bs;
            
            for (ALPHA_INT ai = mat->row_data[br]; ai < mat->row_data[br+1]; ++ai){
                const ALPHA_INT bc = mat->col_data[ai];
                const ALPHA_INT col = bc * bs;
                if(bc > br ){
					continue;
				}
                a0_idx = ai * bs2;
                if(bc == br){
                    for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_row = b_col + 1; b_row < bs; b_row++){
                            
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = ((J *)mat->val_data)[a0_idx + b_row * bs + b_col];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);

                                y[index2(b_row + row,c,ldy)] = alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                y[index2(b_col + col,c,ldy)] = alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                    }
                }
                else{
                    for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
						//dignaol entry A(row+b_row,col+b_col)
                        for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
                            for(ALPHA_INT c = 0 ; c < columns; c++){
                                val_orig = ((J *)mat->val_data)[a0_idx + b_row * bs + b_col];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);
                                
                                y[index2(b_row + row,c,ldy)] = alpha_madde(y[index2(b_row + row,c,ldy)], temp_conj , x[index2(col + b_col,c,ldx)]);
                                y[index2(b_col + col,c,ldy)] = alpha_madde(y[index2(b_col + col,c,ldy)], temp_orig , x[index2(row + b_row,c,ldx)]);
                            }
                        }
                    }
                }
            }
        }
    }
    else return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
