#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t
gemm_csr_col_plain(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            J ctmp;
            ctmp = alpha_setzero(ctmp);
            for (ALPHA_INT ai = mat->row_data[cr]; ai < mat->row_data[cr+1]; ++ai)
            {
                ctmp = alpha_madde(ctmp, ((J*)mat->val_data)[ai], x[index2(cc, mat->col_data[ai], ldx)]);
            }
            y[index2(cc, cr, ldy)] = alpha_mul(y[index2(cc, cr, ldy)], beta);
            y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


