#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <string.h>

template <typename J>
alphasparseStatus_t diagmm_coo_n_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT rowA = mat->rows;
    ALPHA_INT nnz = mat->nnz;

    J diag[rowA];
    
    for(int i = 0; i < rowA; i++)
    {
        alpha_setzero(diag[i]); 
    }
   
    for (ALPHA_INT ar = 0; ar < nnz; ++ar)
    {
        if (mat->col_data[ar] == mat->row_data[ar])
        {
            diag[mat->row_data[ar]] = ((J *)mat->val_data)[ar];
        }
    }

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            J t;
            t = alpha_setzero(t);
            t = alpha_mul(alpha, diag[cr]);
            t = alpha_mul(t, x[index2(cc, cr, ldx)]);
            y[index2(cc, cr, ldy)] = alpha_mul(beta, y[index2(cc, cr, ldy)]);
            y[index2(cc, cr, ldy)] = alpha_add(y[index2(cc, cr, ldy)], t);
            // y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * diag[cr] * x[index2(cc, cr, ldx)];
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
