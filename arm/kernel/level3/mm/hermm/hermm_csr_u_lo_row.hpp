#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"

template <typename J>
alphasparseStatus_t hermm_csr_u_lo_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT r = 0; r < m; ++r)
    {
        for (ALPHA_INT c = 0; c < n; c++)
        {
            J tmp;
            tmp = alpha_mul(alpha, x[index2(r, c, ldx)]);
            y[index2(r, c, ldy)] = alpha_mul(beta, y[index2(r, c, ldy)]);
            y[index2(r, c, ldy)] = alpha_add(y[index2(r, c, ldy)], tmp);
            // y[index2(r, c, ldy)] = y[index2(r, c, ldy)] * beta + x[index2(r, c, ldx)] * alpha;
        }
        for (ALPHA_INT ai = mat->row_data[r]; ai < mat->row_data[r+1]; ++ai)
        {
            ALPHA_INT ac = mat->col_data[ai];
            if (ac < r)
            {
                // J val = alpha * ((J*)mat->val_data)[ai];
                J val;
                J val_c = cmp_conj(((J*)mat->val_data)[ai]);

                val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                val_c = alpha_mul(alpha, val_c);
                for (ALPHA_INT c = 0; c < n; ++c)
                {
                    J tmp;
                    tmp = alpha_mul(val, x[index2(ac, c, ldx)]);
                    y[index2(r, c, ldy)] = alpha_add(y[index2(r, c, ldy)], tmp);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                }                  
                for (ALPHA_INT c = 0; c < n; ++c)
                {
                    J tmp;
                    tmp = alpha_mul(val_c, x[index2(r, c, ldx)]);
                    y[index2(ac, c, ldy)] = alpha_add(y[index2(ac, c, ldy)], tmp);
                    //  y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
                }                   
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
