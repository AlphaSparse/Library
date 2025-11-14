#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_bsr_n_hi_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    const ALPHA_INT bs = A->block_dim;
    J* diag=(J*) alpha_malloc(A->rows*bs*sizeof(J));
    const ALPHA_INT m = A->rows * bs;
    const ALPHA_INT n = A->cols * bs;
    // assert(m==n);
    memset(diag, '\0', m * sizeof(J));

    const ALPHA_INT bs2 = bs * bs;
    const ALPHA_INT b_rows = A->rows;
    const ALPHA_INT b_cols = A->cols;
    const alphasparse_layout_t block_layout = (alphasparse_layout_t)A->block_layout;
    if(block_layout != ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }

    for(ALPHA_INT br = 0 ; br < b_rows; br++){
        for(ALPHA_INT ai = A->row_data[br]; ai < A->row_data[br+1]; ai++){

            ALPHA_INT bc = A->col_data[ai];
            if(bc == br){
                for(ALPHA_INT b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(br,b_row,bs)] = ((J*)A->val_data)[ai * bs2 +  b_row *(bs + 1)];
                }
            }
        }
    }


    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        J* temp = (J*) alpha_malloc(bs*sizeof(J));

        for (ALPHA_INT br = b_rows - 1; br >= 0; br--)
        {
            for(ALPHA_INT i = 0 ; i < bs ; i++){
                temp[i] = alpha_setzero(temp[i]);
            }
            ALPHA_INT diagBlock = -1;
            // memset(temp,'\0', bs * sizeof(J));
            for (ALPHA_INT ai = A->row_data[br]; ai < A->row_data[br+1]; ai++)
            {
                ALPHA_INT bc = A->col_data[ai];
                if(bc > br)
                    //row-major
                    for(ALPHA_INT row = 0; row < bs; row++)
                    {
                    //all entities belongs to upper triangle 
                        ALPHA_INT a0_offset = ai * bs2 +  row * bs;
                        for(ALPHA_INT col = 0 ; col < bs ; col++)
                        {
                            ALPHA_INT y_offset =  (bc * bs + col) * ldy + out_y_col;
                            ALPHA_INT ele_offset =  a0_offset + col;
                            temp[row] = alpha_madde(temp[row], ((J*)A->val_data)[ ele_offset ] ,y[y_offset]);
                        }
                    }
                //diagonal must be none-zero block
                if( bc==br ){
                    diagBlock = ai;
                }
            }
            if(diagBlock == -1)
            {
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            //row-major
            //right-bottom most
            for(ALPHA_INT row = bs - 1; row >=0 ; row--)
            {
                //upper triangle of block
                for(ALPHA_INT col = row + 1 ; col < bs ; col++){
                    ALPHA_INT y_offset =  (br * bs + col) * ldy + out_y_col;
                    temp[row] = alpha_madde(temp[row] ,((J*)A->val_data)[ diagBlock * bs2 +  row * bs + col] ,y[y_offset]);
                }
                
                J t;
                t = alpha_setzero(t);
                t = alpha_mul(alpha,x[(br * bs + row) * ldx + out_y_col] );
                t = alpha_sub(t,temp[row]);
                y[(br * bs + row) * ldy + out_y_col] = alpha_div(t, diag[row + br * bs]);
            }
        }
        alpha_free(temp);

    }
    alpha_free(diag);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
