#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_sky_n_hi_row_conj(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    J diag[m];
    memset(diag, '\0', m * sizeof(J));
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
		diag[r - 1] = cmp_conj(((J*)A->val_data)[indx]);
    }

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r <A->rows; r++)
        {
            J temp = J{};

            ALPHA_INT start = A->pointers[r];
            ALPHA_INT end   = A->pointers[r + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;
            for (ALPHA_INT ai = start; ai < end - 1; ++ai)
            {
                ALPHA_INT c = r - eles_num + idx;
                J cv = ((J*)A->val_data)[ai];
                cv = alpha_conj(cv);
                temp = alpha_madde(temp, cv, y[c * ldy + out_y_col]);
                idx ++;
            }     

            J t;
            t = alpha_mul(alpha, x[r * ldx + out_y_col]);
            t = alpha_sub(t, temp);
            y[r * ldy + out_y_col] = alpha_div(t, diag[r]);
            // y[r * ldy + out_y_col] = (alpha * x[r * ldx + out_y_col] - temp) / diag[r];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
