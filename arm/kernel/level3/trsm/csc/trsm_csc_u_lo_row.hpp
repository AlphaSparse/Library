#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_csc_u_lo_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    #ifdef DEBUG
    printf("trsm_csc_u_lo_row called\n");
    #endif
    
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;

    //initialize y as x*alpha
    for(int i = 0 ; i < m;i++){
        for(int j = 0 ; j < columns ; j++){
            //initialize y[] as x[]*aplha
            y[index2(i,j,ldy)] = alpha_mul(x[index2(i,j,ldx)] ,alpha);
        }
    }

    //csc format, traverse by column
    for(ALPHA_INT c = 0; c < n;++c){
        //following processing simulates Gaussian Elimination 

        //step 1: processing the lower right diagonal ele such that the coefficient equals to 1
        // for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
        //     y[index2(c,out_y_col,ldy)]= alpha_div(y[index2(c,out_y_col,ldy)] ,diag[c]);
        // }
        
        for(ALPHA_INT ai = A->col_data[c]; ai < A->col_data[c+1];ai++){
            ALPHA_INT ar = A->row_data[ai];
            if(c < ar){
        //step 2: use the diagonal ele to eliminate coefficients of other rows at the same column
                for(ALPHA_INT out_y_col = 0; out_y_col < columns;out_y_col++){
                    y[index2(ar,out_y_col,ldy)] = alpha_msube(y[index2(ar,out_y_col,ldy)],((J*)A->val_data)[ai],y[index2(c,out_y_col,ldy)]);
                }
            }
        }
        
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
