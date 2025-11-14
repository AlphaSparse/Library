#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_sky_n_hi_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    J diag[m];
    memset(diag, '\0', m * sizeof(J));
    for (ALPHA_INT r = 0; r < m; r++)
    {
        const ALPHA_INT indx = A->pointers[r + 1] - 1;
        diag[r] = ((J*)A->val_data)[indx];
    }

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
            t = alpha_sub(t, temp);
            y[c * ldy + out_y_col] = alpha_div(t, diag[c]);
            // y[c * ldy + out_y_col] = (alpha * x[c * ldx + out_y_col] - temp) / diag[c];
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
