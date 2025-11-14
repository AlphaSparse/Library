#include"alphasparse/kernel.h"
#include"alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_bsr_n_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT j = 0; j < columns; j++)
        for (ALPHA_INT i = 0; i < mat->rows * mat->block_dim; i++)
            y[i + j * ldy] = alpha_mul(y[i + j * ldy], beta);
            
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

        for(ALPHA_INT c = 0 ; c < columns ; c++){
            for(ALPHA_INT row = 0 ; row < m ; row +=bs ){
                const ALPHA_INT br = row / bs;
                
                for(ALPHA_INT ai= mat->row_data[br]; ai < mat->row_data[br+1]; ++ai){
                    const ALPHA_INT bc = mat->col_data[ai];
                    const ALPHA_INT col = bc * bs;
                    
                    if(bc > br ){
					    continue;
				    }
                    a0_idx = ai * bs2;
                    if(bc == br){
                        for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
						//dignaol entry A(row+b_row,col+b_col)
						temp_orig = alpha_mul(alpha, ((J *)mat->val_data)[a0_idx + b_row  * ( bs+1 ) ]);
						y[index2(c,b_row+row,ldy)] = alpha_madde(y[index2(c,b_row+row,ldy)], temp_orig , x[index2(c,col + b_row,ldx)]);
                            for(ALPHA_INT b_col = 0; b_col < b_row; b_col++){
                                val_orig = ((J *)mat->val_data)[a0_idx + b_row * bs + b_col];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);

                                y[index2(c,b_row + row,ldy)] = alpha_madde(y[index2(c,b_row + row,ldy)], temp_orig , x[index2(c,col + b_col,ldx)]);
                                y[index2(c,b_col + col,ldy)] = alpha_madde(y[index2(c,b_col + col,ldy)], temp_conj , x[index2(c,row + b_row,ldx)]);
                            }
                        }
                    }
                    else{
                        for(ALPHA_INT b_row = 0;b_row < bs; b_row++ ){
                            for(ALPHA_INT b_col = 0; b_col < bs; b_col++){
                                val_orig = ((J *)mat->val_data)[a0_idx + b_row * bs + b_col];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);

                                y[index2(c,b_row + row,ldy)] = alpha_madde(y[index2(c,b_row + row,ldy)], temp_orig , x[index2(c,col + b_col,ldx)]);
                                y[index2(c,b_col + col,ldy)] = alpha_madde(y[index2(c,b_col + col,ldy)], temp_conj , x[index2(c,row + b_row,ldx)]);
                            }
                        }
                    }
                }
            }
        }
    }
    
    else if( mat->block_layout== ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
        for(ALPHA_INT c = 0 ; c < columns ; c++){
            for(int row = 0 ; row < m ; row +=bs ){
                const ALPHA_INT br = row / bs;
                
                for(ALPHA_INT ai= mat->row_data[br]; ai < mat->row_data[br+1]; ++ai){
                    
                    const ALPHA_INT bc = mat->col_data[ai];
                    const ALPHA_INT col = bc * bs;
                    
                    if(bc > br ){
					    continue;
				    }
                    a0_idx = ai * bs2;
                    
                    if(bc == br){
                        for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
                            //dignaol entry A(row+b_row,col+b_col)
                            //y[b_row + row] += alpha*A->values[a0_idx + (b_row + 1) * bs]*x[col + b_col];
                            for(ALPHA_INT b_row = b_col + 1; b_row < bs; b_row++){
                                
                                val_orig = ((J *)mat->val_data)[a0_idx + b_col * bs + b_row];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);
                                y[index2(c,b_row + row,ldy)] = alpha_madde(y[index2(c,b_row + row,ldy)], temp_orig , x[index2(c,col + b_col,ldx)]);
                                y[index2(c,b_col + col,ldy)] = alpha_madde(y[index2(c,b_col + col,ldy)], temp_conj , x[index2(c,row + b_row,ldx)]);	
                            }
                            temp_orig = alpha_mul(alpha, ((J *)mat->val_data)[a0_idx + b_col  * ( bs+1 ) ]);
                            y[index2(c,b_col + row,ldy)] = alpha_madde(y[index2(c,b_col + row,ldy)], temp_orig , x[index2(c,col + b_col,ldx)]);
                        }
                    }
                    else{
                        for(ALPHA_INT b_col = 0;b_col < bs; b_col++ ){
                            for(ALPHA_INT b_row = 0; b_row < bs; b_row++){
                                
                                val_orig = ((J *)mat->val_data)[a0_idx + b_col * bs + b_row];
                                val_conj = cmp_conj(val_orig);
                                temp_orig = alpha_mul(alpha, val_orig);
                                temp_conj = alpha_mul(alpha, val_conj);

                                y[index2(c,b_row + row,ldy)] = alpha_madde(y[index2(c,b_row + row,ldy)], temp_orig , x[index2(c,col + b_col,ldx)]);
                                y[index2(c,b_col + col,ldy)] = alpha_madde(y[index2(c,b_col + col,ldy)], temp_conj , x[index2(c,row + b_row,ldx)]);
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
