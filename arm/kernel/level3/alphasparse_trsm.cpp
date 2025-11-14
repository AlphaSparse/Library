#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "trsm/trsm.h"
// #include "alphasparse/tuning.h"
// #include "alphasparse/inspector.h"

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t trsm_template(const alphasparseOperation_t op_rq,
                          const J alpha,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr dscr_rq, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                          const alphasparse_layout_t layout,      /* storage scheme for the dense matrix: C-style or Fortran-style */
                          const J *x,
                          const ALPHA_INT columns,
                          const ALPHA_INT ldx,
                          J *y,
                          const ALPHA_INT ldy)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    alphasparse_matrix_t compute_mat = NULL;
    struct alpha_matrix_descr compute_descr = dscr_rq;
    alphasparseOperation_t compute_operation = op_rq;

    if (compute_mat == NULL)
        compute_mat = A;
    
    if (compute_mat->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_n_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_n_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_u_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_u_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_n_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_n_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_u_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_u_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_n_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_n_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_u_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_u_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_csr_n_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_csr_u_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_n_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_n_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_u_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_u_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_n_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_n_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_u_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_u_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_n_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_n_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csr_u_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csr_u_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_csr_n_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_csr_u_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_COO)
    {
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_n_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_n_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_u_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_u_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_n_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_n_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_u_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_u_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_n_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_n_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_u_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_u_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_coo_n_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_coo_u_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_n_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_n_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_u_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_u_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_n_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_n_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_u_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_u_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_n_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_n_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_coo_u_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_coo_u_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_coo_n_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_coo_u_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_n_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_n_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_u_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_u_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_n_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_n_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_u_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_u_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_n_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_n_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_u_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_u_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_csc_n_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_csc_u_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_n_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_n_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_u_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_u_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_n_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_n_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_u_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_u_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_n_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_n_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_csc_u_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_csc_u_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_csc_n_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_csc_u_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_n_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_n_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_u_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_u_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_n_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_n_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_u_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_u_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_n_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_n_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_u_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_u_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_bsr_n_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_bsr_u_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_n_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_n_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_u_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_u_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_n_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_n_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_u_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_u_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_n_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_n_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_bsr_u_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_bsr_u_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_bsr_n_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_bsr_u_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_n_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_n_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_u_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_u_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_n_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_n_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_u_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_u_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_n_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_n_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_u_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_u_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_dia_n_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_dia_u_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_n_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_n_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_u_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_u_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_n_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_n_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_u_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_u_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_n_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_n_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_dia_u_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_dia_u_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_dia_n_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_dia_u_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if (compute_mat->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_n_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_n_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_u_lo_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_u_hi_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_n_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_n_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_u_lo_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_u_hi_row_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_n_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_n_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_u_lo_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_u_hi_row_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_sky_n_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_sky_u_row(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
        {
            if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
            {
                if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_n_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_n_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_u_lo_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_u_hi_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_n_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_n_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_u_lo_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_u_hi_col_trans(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
                if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
                {
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_n_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_n_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }    
                    if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    {
                        if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                            return trsm_sky_u_lo_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else if(compute_descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                            return trsm_sky_u_hi_col_conj(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                        else
                            ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                    }                  
                    else
                        ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
                }
            }
            else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
            {
                if(compute_descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                    return diagsm_sky_n_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else if(compute_descr.diag == ALPHA_SPARSE_DIAG_UNIT)
                    return diagsm_sky_u_col(alpha, compute_mat->mat, x, columns, ldx, y, ldy);
                else
                    ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
            {
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
        }
        else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
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
                            TYPE *y,                                    \
                            const ALPHA_INT ldy)                        \
    {                                                                   \
        return trsm_template(op_rq, alpha, A, dscr_rq, layout, x, columns, ldx, y, ldy);\
    }
C_IMPL(alphasparse_s_trsm, float);
C_IMPL(alphasparse_d_trsm, double);
C_IMPL(alphasparse_c_trsm, ALPHA_Complex8);
C_IMPL(alphasparse_z_trsm, ALPHA_Complex16);
#undef C_IMPL