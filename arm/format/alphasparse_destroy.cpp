/**
 * @brief implement for alphasparse_destroy intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

alphasparseStatus_t alphasparse_destroy(alphasparse_matrix_t A)
{
    check_null_return(A, ALPHA_SPARSE_STATUS_SUCCESS);
    if (A->mat != NULL)
    {
            alpha_free(A->mat->row_data);
            alpha_free(A->mat->col_data);
            alpha_free(A->mat->val_data);
            alpha_free(A->mat);
    }
    alpha_free(A);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
