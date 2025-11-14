#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_coo_n_hi_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
#ifdef PRINT
	printf("kernel hermm_c_coo_n_hi_row called\n");
#endif
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;

    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            y[i * ldy + j] = alpha_mul(y[i * ldy + j], beta);

    for (ALPHA_INT ai = 0; ai < mat->nnz; ai++)
    {
        ALPHA_INT ac = mat->col_data[ai];
        ALPHA_INT r = mat->row_data[ai];
        J origin_val = ((J *)mat->val_data)[ai];
        J conj_val = cmp_conj(origin_val);
        J t,t_conj;
        if (ac > r)
        {
            t = alpha_mul(alpha, origin_val);
            t_conj = alpha_mul(alpha, conj_val);

            for (ALPHA_INT c = 0; c < n; ++c)
                y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], t, x[index2(ac, c, ldx)]);
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                y[index2(ac, c, ldy)] = alpha_madde(y[index2(ac, c, ldy)], t_conj, x[index2(r, c, ldx)]);
                // y[index2(ac, c, ldy)] += val * x[index2(r, c, ldx)];
        }
        else if (ac == r)
        {
            // J val = alpha * ((J *)mat->val_data)[ai]; 
            J val;
            val = alpha_mul(alpha, origin_val);
            for (ALPHA_INT c = 0; c < n; ++c)
                y[index2(r, c, ldy)] = alpha_madde(y[index2(r, c, ldy)], val, x[index2(ac, c, ldx)]);
                // y[index2(r, c, ldy)] += val * x[index2(ac, c, ldx)];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
