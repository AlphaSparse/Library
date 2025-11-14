#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "memory.h"

template <typename J>
alphasparseStatus_t diagmm_csr_n_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    J *diag = (J *)alpha_malloc(mat->rows * sizeof(J));
    for (ALPHA_INT ar = 0; ar < mat->rows; ++ar)
    {
        diag[ar] = alpha_setzero(diag[ar]);
        for (ALPHA_INT ai = mat->row_data[ar]; ai < mat->row_data[ar+1]; ++ai)
            if (mat->col_data[ai] == ar)
            {
                diag[ar] = ((J*)mat->val_data)[ai];
            }
    }

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            J val;
            y[index2(cc, cr, ldy)] = alpha_mul(y[index2(cc, cr, ldy)], beta);
            val = alpha_mul(alpha, diag[cr]);
            y[index2(cc, cr, ldy)] = alpha_madd(val, x[index2(cc, cr, ldx)], y[index2(cc, cr, ldy)]);
        }

    alpha_free(diag);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
