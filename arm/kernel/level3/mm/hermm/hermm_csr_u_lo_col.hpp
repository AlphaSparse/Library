#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"

template <typename J>
alphasparseStatus_t hermm_csr_u_lo_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
        {
            J t = J{};
            y[i + j * ldy] = alpha_mul(beta, y[i + j * ldy]);
            t = alpha_mul(alpha, x[i + j * ldx]);
            y[i + j * ldy] = alpha_add(y[i + j * ldy], t);
            // y[i] = beta * y[i] + alpha * x[i];
        }
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = 0; cr < mat->rows; ++cr)
        {
            for (ALPHA_INT ai = mat->row_data[cr]; ai < mat->row_data[cr+1]; ++ai)
            {
                ALPHA_INT ac = mat->col_data[ai];
                if (ac < cr)
                {
                    J tmp, tmp2;
                    J val_c = cmp_conj(((J*)mat->val_data)[ai]);

                    tmp = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                    tmp2 = alpha_mul(tmp, x[index2(cc, ac, ldx)]);
                    y[index2(cc, cr, ldy)] = alpha_add(y[index2(cc, cr, ldy)], tmp2);
                    tmp = alpha_mul(alpha, val_c);
                    tmp2 = alpha_mul(tmp, x[index2(cc, cr, ldx)]);
                    y[index2(cc, ac, ldy)] = alpha_add(y[index2(cc, ac, ldy)], tmp2);
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
