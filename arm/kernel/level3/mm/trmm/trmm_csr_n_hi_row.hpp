#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

#include <memory.h>

template <typename J>
alphasparseStatus_t trmm_csr_n_hi_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{

    for (ALPHA_INT r = 0; r < mat->rows; ++r)
    {
        J *Y = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < columns; c++)
            Y[c] = alpha_mul(Y[c], beta);
        for (ALPHA_INT ai = mat->row_data[r]; ai < mat->row_data[r+1]; ai++)
        {
            ALPHA_INT ac = mat->col_data[ai];
            if (ac >= r)
            {
                J val;
                val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                const J *X = &x[index2(ac, 0, ldx)];
                for (ALPHA_INT c = 0; c < columns; ++c)
                    Y[c] = alpha_madd(val, X[c], Y[c]);
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
