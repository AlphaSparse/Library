#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_csc_u_hi_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    #ifdef DEBUG
    printf("trsm_csc_u_hi_col called\n");
    #endif
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
//assume m==n 
    for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
        for(int i = 0 ; i < n ; i++){
            //initialize y[] as x[]*aplha
            y[index2(out_y_col,i,ldy)] = alpha_mul(alpha, x[index2(out_y_col,i,ldx)]);
        }
        for(ALPHA_INT c = n - 1; c >= 0;--c){//csc format, traverse by column
            for(ALPHA_INT ai = A->col_data[c+1]-1; ai >= A->col_data[c];ai--){
                ALPHA_INT ar = A->row_data[ai];
                if(ar < c){
                    y[index2(out_y_col,ar,ldy)] = alpha_msube(y[index2(out_y_col,ar,ldy)], ((J*)A->val_data)[ai], y[index2(out_y_col,c,ldy)]);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
