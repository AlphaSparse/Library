#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/spdef.h"
#include "alphasparse/inspector.h"
#include "mm/gemm/gemm.h"
#include "mm/diagmm/diagmm.h"
#include "mm/hermm/hermm.h"
#include "mm/symm/symm.h"
#include "mm/trmm/trmm.h"

// #include "alphasparse/tuning.h"

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t alphasparse_mm_template(const alphasparseOperation_t op_rq, // operation_request
                          const J alpha,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr dscr_rq, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                          const alphasparse_layout_t layout,      /* storage scheme for the dense matrix: C-style or Fortran-style */
                          const J *x,
                          const ALPHA_INT columns,
                          const ALPHA_INT ldx,
                          const J beta,
                          J *y,
                          const ALPHA_INT ldy)
{
    check_null_return(A, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    // check_return(!check_data_type(A->datatype_cpu), ALPHA_SPARSE_STATUS_INVALID_VALUE);

    alphasparse_matrix_t compute_mat = NULL;
    struct alpha_matrix_descr compute_descr = dscr_rq;
    alphasparseOperation_t compute_operation = op_rq;
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC || compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        // Check if it is a square matrix
        check_return(A->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if (compute_mat == NULL)
        compute_mat = A;

    if (compute_mat->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_csr_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_csr_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_csr_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_csr_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_csr_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_csr_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_csr_n_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_csr_n_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_csr_u_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_csr_u_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csr_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csr_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csr_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csr_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csr_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csr_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_COO)
    {
        if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_coo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_coo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_coo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_coo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_coo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_coo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_coo_n_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_coo_n_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_coo_u_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_coo_u_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_coo_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_coo_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_coo_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_coo_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_coo_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_coo_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_csc_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_csc_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_csc_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_csc_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_csc_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_csc_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_csc_n_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_csc_n_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_csc_u_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_csc_u_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_csc_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_csc_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_csc_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_csc_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_csc_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_csc_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_bsr_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_bsr_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_bsr_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_bsr_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_bsr_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_bsr_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_bsr_n_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_bsr_n_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_bsr_u_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_bsr_u_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_bsr_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_bsr_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_bsr_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_bsr_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_bsr_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_bsr_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_dia_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_dia_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_dia_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_dia_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return gemm_dia_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return gemm_dia_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_dia_n_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_dia_n_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_dia_u_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_dia_u_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_dia_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_dia_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_dia_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_dia_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_dia_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_dia_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_sky_n_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_sky_n_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
            {
                if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    return diagmm_sky_u_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    return diagmm_sky_u_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return hermm_sky_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return hermm_sky_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return symm_sky_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return symm_sky_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
        else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {
            if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_n_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_n_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_n_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_n_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_u_lo_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_u_hi_row<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_u_lo_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_u_hi_col<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_n_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_n_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_n_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_n_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_u_lo_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_u_hi_row_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_u_lo_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_u_hi_col_trans<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
            {
                if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_n_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_n_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_n_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_n_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
                {
                    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_u_lo_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_u_hi_row_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                    {
                        if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trmm_sky_u_lo_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trmm_sky_u_hi_col_conj<J>(alpha, compute_mat->mat, x, columns, ldx, beta, y, ldy);
                        else
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                    }
                    else
                        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                }
                else
                    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
            }
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
        }
    }
    else
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define C_IMPL(ONAME, TYPE)                                             \
    alphasparseStatus_t ONAME(const alphasparseOperation_t op_rq,       \
                            const TYPE alpha,                           \
                            const alphasparse_matrix_t A,               \
                            const struct alpha_matrix_descr dscr_rq,    \
                            const alphasparse_layout_t layout,          \
                            const TYPE *x,                              \
                            const ALPHA_INT columns,                    \
                            const ALPHA_INT ldx,                        \
                            const TYPE beta,                            \
                            TYPE *y,                                    \
                            const ALPHA_INT ldy)                        \
    {                                                                   \
        return alphasparse_mm_template(op_rq,                           \
                                       alpha,                           \
                                       A,                               \
                                       dscr_rq,                         \
                                       layout,                          \
                                       x,                               \
                                       columns,                         \
                                       ldx,                             \
                                       beta,                            \
                                       y,                               \
                                       ldy);                            \
    }

C_IMPL(alphasparse_s_mm, float);
C_IMPL(alphasparse_d_mm, double);
C_IMPL(alphasparse_c_mm, ALPHA_Complex8);
C_IMPL(alphasparse_z_mm, ALPHA_Complex16);
#undef C_IMPL