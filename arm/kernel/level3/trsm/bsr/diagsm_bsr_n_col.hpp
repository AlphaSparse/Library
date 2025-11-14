#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagsm_bsr_n_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
#ifdef DEBUG
    printf("kernel diagsm_bsr_n_col called \n");
#endif
    const ALPHA_INT bs = A->block_dim;
    J* diag=(J*) alpha_malloc(A->rows*bs*sizeof(J));
    const ALPHA_INT m = A->rows * bs;
    const ALPHA_INT n = A->cols * bs;
    // assert(m==n);
    memset(diag, '\0', m * sizeof(J));
    
    const ALPHA_INT b_rows = A->rows;
    const ALPHA_INT b_cols = A->cols;

    for(ALPHA_INT r = 0 ; r < b_rows; r++){
        for(ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++){
            
            ALPHA_INT ac = A->col_data[ai];
            if(ac == r){
                for(ALPHA_INT b_row = 0 ; b_row < bs ; b_row++){
                    diag[index2(r,b_row,bs)] = ((J*)A->val_data)[ai * bs * bs +  b_row *(bs + 1)];

                }
            }
        }
    }
    
    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows * bs; ++r)
        {
            J t;
            t = alpha_mul(alpha, x[index2(c, r, ldx)]);
            y[index2(c, r, ldy)] = alpha_div(t, diag[r]);
        }
    }

    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
