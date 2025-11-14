#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_sky_n_hi_col_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            y[index2(j, i, ldy)] = alpha_mul(y[index2(j, i, ldy)], beta);

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
                    J tmp_c;
                    tmp_c = cmp_conj(((J *)mat->val_data)[ai]);
                    tmp_c = alpha_mul(alpha, tmp_c);
                    tmp = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                    y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], tmp_c, x[index2(cc, ac, ldx)]);
                    y[index2(cc, ac, ldy)] = alpha_madde(y[index2(cc, ac, ldy)], tmp, x[index2(cc, cr, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, ac, ldx)];
                    // y[index2(cc, ac, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, cr, ldx)];
                }
                else if (ac == cr)
                {
                    //J tmp;
                    J tmp_c;
                    tmp_c = cmp_conj(((J *)mat->val_data)[ai]);
                    tmp_c = alpha_mul(alpha, tmp_c);
                    //tmp = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                    y[index2(cc, cr, ldy)] = alpha_madde(y[index2(cc, cr, ldy)], tmp_c, x[index2(cc, ac, ldx)]);
                    // y[index2(cc, cr, ldy)] += alpha * ((J *)mat->val_data)[ai] * x[index2(cc, ac, ldx)];
                }
                idx ++;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
