#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_sky_n_hi_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            y[index2(i, j, ldy)] = alpha_mul(y[index2(i, j, ldy)], beta);

    for (ALPHA_INT ac = 0; ac < mat->cols; ++ac)
    {
        ALPHA_INT start = mat->pointers[ac];
        ALPHA_INT end   = mat->pointers[ac + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;
        for (ALPHA_INT ai = start; ai < end; ai++)
        {
            ALPHA_INT r = ac - eles_num + idx;
            if (ac > r)
            {
                J val;
                J val_c;
                val_c = cmp_conj(((J *)mat->val_data)[ai]);
                val_c = alpha_mul(alpha, val_c);
                val = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                for (ALPHA_INT c = 0; c < n; ++c)
                    y[index2(ac, c, ldy)] = alpha_madde(y[index2(ac, c, ldy)], val_c, x[index2(r, c, ldx)]);
                    // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
            }
            else if(ac == r)
            {
                J val;
                val = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            }
            idx ++;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
