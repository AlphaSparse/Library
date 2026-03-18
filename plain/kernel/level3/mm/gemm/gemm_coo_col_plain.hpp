#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"

template <typename TYPE>
alphasparseStatus_t
gemm_coo_col_plain(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT i = 0; i < mat->rows; i++)
        for(ALPHA_INT j = 0; j < columns; j++)
            y[i + j * ldy] = alpha_mul(y[i + j * ldy], beta);

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT nn = 0; nn < mat->nnz; ++nn)
        {
            TYPE ctmp;
            ctmp = alpha_mul(((TYPE*)mat->val_data)[nn], x[index2(cc, mat->col_data[nn], ldx)]); 
            y[index2(cc, mat->row_data[nn], ldy)] = alpha_madde(y[index2(cc, mat->row_data[nn], ldy)], alpha, ctmp);
            // y[index2(mat->row_indx[nn], cc, ldy)] += alpha * mat->values[nn] * x[index2(mat->col_indx[nn], cc, ldx)];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
