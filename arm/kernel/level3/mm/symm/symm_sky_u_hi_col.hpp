#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_sky_u_hi_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
        {
            J t;
            y[i + j * ldy] = alpha_mul(beta, y[i + j * ldy]);
            t = alpha_mul(alpha, x[i + j * ldx]);
            y[i + j * ldy] = alpha_add(y[i + j * ldy], t);
            // y[i + j * ldy] = beta * y[i + j * ldy] + alpha * x[i + j * ldx];
        }            
            
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT ac = 0; ac < mat->cols; ++ac)
        {
            ALPHA_INT start = mat->pointers[ac];
            ALPHA_INT end   = mat->pointers[ac + 1];
            ALPHA_INT idx = 1;
            ALPHA_INT eles_num = end - start;

            for (ALPHA_INT ai = start; ai < end; ++ai)
            {
                ALPHA_INT cr = ac - eles_num + idx;
                if (ac > cr)
                {
                    J tmp;
                    tmp = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                    y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], tmp, x[index2(cc, ac, ldx)]);
                    y[index2(cc, ac, ldy)] = alpha_madde(y[index2(cc, ac, ldy)], tmp, x[index2(cc, cr, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, ac, ldx)];
                    // y[index2(cc, ac, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, cr, ldx)];
                }
                idx ++;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
