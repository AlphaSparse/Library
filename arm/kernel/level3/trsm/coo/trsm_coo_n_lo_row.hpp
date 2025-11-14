#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t trsm_coo_n_lo_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = A->rows;
    J diag[m];
    memset(diag, '\0', m * sizeof(J));
    for (ALPHA_INT r = 0; r < A->nnz; r++)
    {
        if(A->row_data[r] == A->col_data[r])
        {
            diag[A->row_data[r]] = ((J*)A->val_data)[r];
        }
    }

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
                    {temp = alpha_madde(temp, ((J*)A->val_data)[cr], y[col * ldy + out_y_col]);}
                    // temp += ((J*)A->val_data)[cr] * y[col * ldy + out_y_col];
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
