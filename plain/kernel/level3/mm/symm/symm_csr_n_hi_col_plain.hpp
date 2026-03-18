#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/bisearch.h"

template <typename J>
alphasparseStatus_t
symm_csr_n_hi_col_plain(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT c = 0; c < columns; c++)
        for (ALPHA_INT r = 0; r < mat->rows; r++)
            y[index2(c, r, ldy)] = alpha_mul(y[index2(c, r, ldy)], beta);

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT ar = 0; ar < mat->rows; ++ar)
        {
            for (ALPHA_INT ai = mat->row_data[ar]; ai < mat->row_data[ar+1]; ++ai)
            {
                ALPHA_INT ac = mat->col_data[ai];
                if (ac > ar)
                {
                    J val;
                    val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                    y[index2(cc, ar, ldy)] = alpha_madde(y[index2(cc, ar, ldy)], val, x[index2(cc, ac, ldx)]);
                    y[index2(cc, ac, ldy)] = alpha_madde(y[index2(cc, ac, ldy)], val, x[index2(cc, ar, ldx)]);
                }
                else if (ac == ar)
                {
                    J val;
                    val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                    y[index2(cc, ar, ldy)] = alpha_madde(y[index2(cc, ar, ldy)], val, x[index2(cc, ac, ldx)]);
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
