#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_sky_n_lo_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    J diag[m];
    memset(diag, '\0', m * sizeof(J));
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
        diag[r - 1] = ((J*)A->val_data)[indx];
    }

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r <A->rows; r++)
        {
            J temp;
            temp = alpha_setzero(temp);

            ALPHA_INT start = A->pointers[r];
            ALPHA_INT end   = A->pointers[r + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end - 1; ++ai)
            {
                ALPHA_INT c = r - eles_num + idx;
                temp = alpha_madde(temp, ((J*)A->val_data)[ai], y[out_y_col * ldy + c]);
                idx ++;
            }     

            J t;
            t = alpha_mul(alpha, x[out_y_col * ldx + r]);
            t = alpha_sub(t, temp);
            y[out_y_col * ldy + r] = alpha_div(t, diag[r]);
            // y[out_y_col * ldy + r] = (alpha * x[out_y_col * ldx + r] - temp) / diag[r];
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
