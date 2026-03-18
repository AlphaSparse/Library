#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include "alphasparse/util/partition.h"
#include "alphasparse/compute.h"
#include <type_traits>

template <typename J>
alphasparseStatus_t gemm_csr_row_plain(const J alpha, const internal_spmat mat, const J *x,
                          const ALPHA_INT columns, const ALPHA_INT ldx, const J beta,
                          J *y, const ALPHA_INT ldy) {
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = columns;
    for (ALPHA_INT r = 0; r < m; ++r)
    {
        J *Y = &y[index2(r, 0, ldy)];
        for (ALPHA_INT c = 0; c < n; c++)
            Y[c] = alpha_mul(Y[c], beta);
        for (ALPHA_INT ai = mat->row_data[r]; ai < mat->row_data[r+1]; ai++)
        {
            J val;
            val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
            const J *X = &x[index2(mat->col_data[ai], 0, ldx)];
            for (ALPHA_INT c = 0; c < n; ++c)
                Y[c] = alpha_madde(Y[c], val, X[c]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
