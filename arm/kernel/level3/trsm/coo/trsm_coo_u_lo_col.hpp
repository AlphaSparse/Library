#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_coo_u_lo_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {        
        for (ALPHA_INT r = 0; r < m; r++)
        {
            J temp;// = {.real = 0.0f, .imag = 0.0f};
            temp = alpha_setzero(temp);
            for (ALPHA_INT cr = 0; cr < A->nnz; cr++)
            {
                int row = A->row_data[cr];
                int col = A->col_data[cr];
                if(row == r && col < r)
                    {temp = alpha_madde(temp, ((J*)A->val_data)[cr], y[out_y_col * ldy + col]);}
                    // temp += ((J*)A->val_data)[cr] * y[out_y_col * ldy + col];
            }
            J t;
            t = alpha_mul(alpha, x[out_y_col * ldx + r]);
            y[out_y_col * ldy + r] = alpha_sub(t, temp);
            // y[out_y_col * ldy + r] = (alpha * xl[out_y_col * ldx + r] - temp) ;
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
