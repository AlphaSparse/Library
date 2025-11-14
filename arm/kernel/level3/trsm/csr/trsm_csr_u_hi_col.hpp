#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"

template <typename J>
alphasparseStatus_t trsm_csr_u_hi_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = m - 1; r >= 0; r--)
        {
            J temp;
            temp = alpha_setzero(temp);
            for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
            {
                ALPHA_INT ac = A->col_data[ai];
                if (ac > r)
                {
                    temp = alpha_madd(((J*)A->val_data)[ai], y[out_y_col * ldy + ac], temp);
                }
            }
            J t;
            t = alpha_setzero(t);
            t = alpha_mul(alpha, x[out_y_col * ldx + r]);
            y[out_y_col * ldy + r] = alpha_sub(t, temp);
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
