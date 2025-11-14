#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagsm_csc_n_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{//assume A is square
    J* diag=(J*) alpha_malloc(A->rows*sizeof(J));

    memset(diag, '\0', A->rows * sizeof(J));

    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        for (ALPHA_INT ai = A->col_data[c]; ai < A->col_data[c+1]; ai++)
        {
            ALPHA_INT ar = A->row_data[ai];
            if (ar == c)
            {
                diag[c] = ((J*)A->val_data)[ai];

                //break;
            }
        }
    }

    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows; ++r)
        {
            J t;
            t = alpha_mul(alpha, x[index2(c, r, ldx)]);
            y[index2(c, r, ldy)] = alpha_div(t, diag[r]);
        }
    }
    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
