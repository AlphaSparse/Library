#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename TYPE>
alphasparseStatus_t
gemm_coo_col_transXY(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    ALPHA_INT ldX = columns;
    ALPHA_INT ldY = columns;
    TYPE *X_ = (TYPE *)alpha_malloc(mat->cols * ldX * sizeof(TYPE));
    TYPE *Y_ = (TYPE *)alpha_malloc(mat->rows * ldY * sizeof(TYPE));

    pack_matrix_col2row(mat->cols, columns, x, ldx, X_, ldX);
    pack_matrix_col2row(mat->rows, columns, y, ldy, Y_, ldY);
    gemm_coo_row(alpha, mat, X_, columns, ldX, beta, Y_, ldY); //alpha,mat,X,columns,ldX,beta,Y,ldY

    pack_matrix_row2col(mat->rows, columns, Y_, ldY, y, ldy);
    alpha_free(X_);
    alpha_free(Y_);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t
gemm_coo_col(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    gemm_coo_col_transXY(alpha, mat, x, columns, ldx, beta, y, ldy);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
