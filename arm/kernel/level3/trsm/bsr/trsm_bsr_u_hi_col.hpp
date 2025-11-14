#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_bsr_u_hi_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    const ALPHA_INT bs = A->block_dim;
    const ALPHA_INT bs2 = bs * bs;
    const ALPHA_INT m = A->rows * bs;
    const ALPHA_INT n = A->cols * bs;
    const ALPHA_INT b_rows = A->rows;
    const ALPHA_INT b_cols = A->cols;
    const alphasparse_layout_t block_layout = (alphasparse_layout_t)A->block_layout;
    if(block_layout != ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        J* temp = (J*) alpha_malloc(bs*sizeof(J));
        const ALPHA_INT y0_offset = out_y_col * ldy;
        const ALPHA_INT x0_offset = out_y_col * ldx;

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
                    //col-major
                    for(ALPHA_INT col = 0; col < bs; col++)
                    {
                    //all entities belongs to upper triangle 
                        ALPHA_INT y_offset =  y0_offset + bc * bs + col;
                        ALPHA_INT a0_offset = ai * bs2 +  col * bs;
                        for(ALPHA_INT row = 0 ; row < bs ; row++)
                        {
                            
                            ALPHA_INT ele_offset =  a0_offset + row;
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
            //col-major
            //right-bottom most
            for(ALPHA_INT col = bs - 1; col >= 0; col--)
            {
                //upper triangle of block
                J t;
                t = alpha_setzero(t);
                t = alpha_mul(alpha,x[x0_offset + br * bs + col]);
                y[y0_offset + br * bs + col] = alpha_sub(t,temp[col]);

                for(ALPHA_INT row = col - 1; row >= 0; row--){
                    temp[row] = alpha_madde(temp[row], ((J*)A->val_data)[ diagBlock * bs2 +  col * bs + row],y[y0_offset + br * bs + col ]);
                }
            }
        }
        alpha_free(temp);
        
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
