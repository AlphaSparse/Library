#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trmm_coo_u_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        J ctmp;
        ctmp = alpha_setzero(ctmp);
        ALPHA_INT r = 0;
        for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
        {
            ALPHA_INT cr =  mat->row_data[nn]; 
            while(cr >= r)
            {
                J t;
                t = alpha_setzero(t);
                y[index2(cc, r, ldy)] = alpha_mul(beta, y[index2(cc, r, ldy)]);
                t = alpha_mul(alpha, x[index2(cc, r, ldx)]);
                y[index2(cc, r, ldy)] = alpha_add(y[index2(cc, r, ldy)], t);
                // y[index2(cc, r, ldy)] = beta * y[index2(cc, r, ldy)] + alpha * x[index2(cc, r, ldx)]; 
                r++;
            }   
            if(mat->col_data[nn] < cr)    
            {    
                ctmp = alpha_madde(ctmp, ((J *)mat->val_data)[nn], x[index2(cc, mat->col_data[nn], ldx)]);
                // ctmp += ((J *)mat->val_data)[nn] * x[index2(cc, mat->col_data[nn], ldx)];                 
            }
            if(nn + 1 < mat->nnz && cr != mat->row_data[nn + 1])
            {
                y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
                // y[index2(cc, cr, ldy)] += alpha * ctmp;
                ctmp = alpha_setzero(ctmp);
            }
            else if(nn + 1 == mat->nnz)
                y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], alpha, ctmp);
        }
        while(mat->rows > r)
        {
            y[index2(cc, r, ldy)] = alpha_mul(beta, y[index2(cc, r, ldy)]);
            // y[index2(cc, r, ldy)] = beta * y[index2(cc, r, ldy)];
            r++;
        } 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
