#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_sky_u_hi_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
        {
            J temp;
            temp = alpha_setzero(temp);
            for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
            {
                ALPHA_INT start = A->pointers[ic];
                ALPHA_INT end   = A->pointers[ic + 1];
                ALPHA_INT eles_num = ic - c;
                if(end - eles_num - 1 >= start)
                    temp = alpha_madde(temp, ((J*)A->val_data)[end - eles_num - 1], y[ic * ldy + out_y_col]);
            }

            J t;
            t = alpha_mul(alpha, x[c * ldx + out_y_col]);
            y[c * ldy + out_y_col] = alpha_sub(t, temp);
            // y[c * ldy + out_y_col] = alpha * x[c * ldx + out_y_col] - temp;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
