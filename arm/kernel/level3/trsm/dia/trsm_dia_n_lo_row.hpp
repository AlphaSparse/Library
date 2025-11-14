#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_dia.hpp"
#include "format/transpose_conj_dia.hpp"
#include "format/destroy_dia.hpp"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_dia_n_lo_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    ALPHA_INT main_diag_pos = 0;
    J diag[m];
    memset(diag, '\0', m * sizeof(J));
    
    for (ALPHA_INT i = 0; i < A->ndiag; i++)
    {
        if(A->dis_data[i] == 0)
        {
            main_diag_pos = i;
            for (ALPHA_INT r = 0; r < A->rows; r++)
            {
                diag[r] = ((J*)A->val_data)[i * A->lval + r];
            }
            break;
        }
    }

    for(ALPHA_INT out_y_col = 0; out_y_col < columns; out_y_col++)
    {
        for (ALPHA_INT r = 0; r < m; r++)
        {
            J temp;
            temp = alpha_setzero(temp);
            for (ALPHA_INT ndiag = 0; ndiag < main_diag_pos; ndiag++)
            {
                if (-A->dis_data[ndiag] <= r)
                {
                    ALPHA_INT ac = r + A->dis_data[ndiag];
                    temp = alpha_madde(temp, ((J*)A->val_data)[ndiag * A->lval + r], y[ac * ldy + out_y_col]);
                    // temp += ((J*)A->val_data)[ndiag * A->lval + r] * y[ac * ldy + out_y_col];
                }
            }
            J t;
            t = alpha_setzero(t);
            t = alpha_mul(alpha, x[r * ldx + out_y_col]);
            t = alpha_sub(t, temp);
            y[r * ldy + out_y_col] = alpha_div(t, diag[r]);
            // y[r * ldy + out_y_col] = (alpha * x[r * ldx + out_y_col] - temp) / diag[r];
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
