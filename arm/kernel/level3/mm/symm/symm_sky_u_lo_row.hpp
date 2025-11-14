#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t symm_sky_u_lo_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
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

        ALPHA_INT start = mat->pointers[r];
        ALPHA_INT end   = mat->pointers[r + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;

        for (ALPHA_INT ai = start; ai < end; ai++)
        {
            ALPHA_INT ac = r - eles_num + idx;
            if (ac < r)
            {
                J val;
                val = alpha_mul(alpha, ((J *)mat->val_data)[ai]);
                for (ALPHA_INT c = 0; c < n; ++c)
                    y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                    // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
                for (ALPHA_INT c = 0; c < n; ++c)
                    y[index2(ac, c, ldy)] = alpha_madde(y[index2(ac, c, ldy)], val, x[index2(r, c, ldx)]);
                    // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
            }
            idx++;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
