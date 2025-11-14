#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t
symm_coo_u_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
        {
            J t;
            t = alpha_setzero(t);
            y[i + j * ldy] = alpha_mul(beta, y[i + j * ldy]);
            t = alpha_mul(alpha, x[i + j * ldx]);
            y[i + j * ldy] = alpha_add(y[i + j * ldy], t);
            // y[i] = beta * y[i] + alpha * x[i];
        }
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT ai = 0; ai < mat->nnz; ++ai)
        {
            ALPHA_INT ac = mat->col_data[ai];
            ALPHA_INT cr = mat->row_data[ai];
            if (ac < cr)
            {
                J t;
                t = alpha_setzero(t);
                t = alpha_mul(((J *)mat->val_data)[ai], alpha);
                y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], t, x[index2(cc, ac, ldx)]);
                y[index2(cc, ac, ldy)] = alpha_madde(y[index2(cc, ac, ldy)], t, x[index2(cc, cr, ldx)]);
                // y[index2(cc, cr, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, ac, ldx)];
                // y[index2(cc, ac, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, cr, ldx)];
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
