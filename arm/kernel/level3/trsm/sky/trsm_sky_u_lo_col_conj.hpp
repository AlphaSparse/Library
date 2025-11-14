#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_sky_u_lo_col_conj(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
        {
            J temp = J{};
            for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
            {
                ALPHA_INT start = A->pointers[ic];
                ALPHA_INT end   = A->pointers[ic + 1];
                ALPHA_INT eles_num = ic - c;
                if(end - eles_num - 1 >= start)
                {
                    J cv = ((J*)A->val_data)[end - eles_num - 1];
                    cv = alpha_conj(cv);
                    temp = alpha_madde(temp, cv, y[out_y_col * ldy + ic]);
                }
            }

            J t;
            t = alpha_mul(alpha, x[out_y_col * ldx + c]);
            y[out_y_col * ldy + c] = alpha_sub(t, temp);
            // y[out_y_col * ldy + c] = alpha * x[out_y_col * ldx + c] - temp;
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
