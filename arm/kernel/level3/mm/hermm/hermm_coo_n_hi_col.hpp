#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_coo_n_hi_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for(ALPHA_INT j = 0; j < columns; j++)
        for (ALPHA_INT i = 0; i < mat->rows; i++)
            y[i + j * ldy] = alpha_mul(y[i + j * ldy], beta);
        // y[i] *= beta;
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT nnz = 0; nnz < mat->nnz; ++nnz)
        {
            ALPHA_INT cr = mat->row_data[nnz];
            ALPHA_INT ac = mat->col_data[nnz];
            // printf("get A (%d,%d):(%.10f,%.10f)\n",cr,ac,((J *)mat->val_data)[nnz].real,((J *)mat->val_data)[nnz].imag);

            if (ac > cr)
            {
                J t,t_conj;
                J origin_val = ((J *)mat->val_data)[nnz];
                J conj_val;
                conj_val = alpha_conj(origin_val);
                
                t = alpha_mul(origin_val, alpha);
                t_conj = alpha_mul(conj_val, alpha);

                y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)] , t, x[index2(cc, ac, ldx)]);
                y[index2(cc, ac,  ldy)] = alpha_madde(y[index2(cc, ac,  ldy)], t_conj, x[index2(cc, cr, ldx)]);
            }
            else if (ac == cr)
            {
                J t;
                t = alpha_setzero(t);
                t = alpha_mul(((J *)mat->val_data)[nnz], alpha);
                y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], t, x[index2(cc, ac, ldx)]);
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
