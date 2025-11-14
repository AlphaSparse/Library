#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_sky_u_hi_row_conj(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
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
            y[r * ldy + out_y_col] = alpha_sub(t, temp);
            // y[r * ldy + out_y_col] = alpha * x[r * ldx + out_y_col] - temp;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
