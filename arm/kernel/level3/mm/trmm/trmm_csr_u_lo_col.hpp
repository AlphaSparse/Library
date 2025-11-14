#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"


template <typename J>
alphasparseStatus_t trmm_csr_u_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            y[index2(cc, cr, ldy)] = alpha_mul(y[index2(cc, cr, ldy)], beta);
            y[index2(cc, cr, ldy)] = alpha_madd(alpha, x[index2(cc, cr, ldx)], y[index2(cc, cr, ldy)]);
            J ctmp;
            ctmp = alpha_setzero(ctmp);
            for (ALPHA_INT ai = mat->row_data[cr]; ai < mat->row_data[cr+1]; ++ai)
            {
                ALPHA_INT ac = mat->col_data[ai];
                if (ac < cr)
                {
                    ctmp = alpha_madd(((J*)mat->val_data)[ai], x[index2(cc, ac, ldx)], ctmp);
                }
            }
            y[index2(cc, cr, ldy)] = alpha_madd(alpha, ctmp, y[index2(cc, cr, ldy)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
