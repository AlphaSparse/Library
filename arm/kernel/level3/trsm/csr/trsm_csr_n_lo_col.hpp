#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_csr_n_lo_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    J diag[m];
    memset(diag, '\0', m * sizeof(J));
    for (ALPHA_INT r = 0; r < m; r++)
    {
        for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
        {
            ALPHA_INT ac = A->col_data[ai];
            if (ac == r)
            {
                diag[r] = ((J*)A->val_data)[ai];
            }
        }
    }

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r < m; r++)
        {
            J temp;
            temp = alpha_setzero(temp);
            for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
            {
                ALPHA_INT ac = A->col_data[ai];
                if (ac < r)
                {
                    temp = alpha_madd(((J*)A->val_data)[ai], y[out_y_col * ldy + ac], temp);
                }
            }
            J t;
            t = alpha_setzero(t);
            t = alpha_mul(alpha, x[out_y_col * ldx + r]);
            t = alpha_sub(t, temp);
            y[out_y_col * ldy + r] = alpha_div(t, diag[r]);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
