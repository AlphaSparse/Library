/**
 * @brief implement for alphasparse_?_trsv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/spdef.h"
#include "alphasparse/util/internal_check.h"
#include <cstdio>

#include "trsv/spsv_plain.h"
 /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t alphasparse_trsv_template_plain(const alphasparseOperation_t operation, 
                          const J alpha, 
                          const alphasparse_matrix_t A, 
                          const struct alpha_matrix_descr descr,
                         const J *x,
                         J *y)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    // check_return(A->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);

// #ifndef COMPLEX
//     if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
// #endif

    check_return(A->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    if(A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        //TRIANG
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {   
            //not trans
            if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csr_n_hi_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, y);
                    else
                        return trsv_csr_u_hi_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csr_n_lo_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, y);
                    else
                        return trsv_csr_u_lo_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT){

                        return trsv_csr_n_hi_trans_plain(alpha, A->mat, x, y);
                    }
                     else{
                        return trsv_csr_u_hi_trans_plain(alpha, A->mat, x, y);
                    }
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT){
                        return trsv_csr_n_lo_trans_plain(alpha, A->mat, x, y);
                    }
                    else{
                        return trsv_csr_u_lo_trans_plain(alpha, A->mat, x, y);
                    }     
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT){
                        return trsv_csr_n_hi_conj_plain(alpha, A->mat, x, y);
                    }
                     else{
                        return trsv_csr_u_hi_conj_plain(alpha, A->mat, x, y);
                    }
                }else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT){
                        return trsv_csr_n_lo_conj_plain(alpha, A->mat, x, y);
                    }
                    else{
                        return trsv_csr_u_lo_conj_plain(alpha, A->mat, x, y);
                    }    

                }
            }
            else
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT){
                return diagsv_csr_n_plain(alpha, A->mat, x, y);
            }
            else{
                return diagsv_csr_u_plain(alpha, A->mat, x, y);
            }
        }else
        {
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        //TRIANG
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {   
            //not trans
            if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_coo_n_hi_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->col_data, (J*)A->mat->val_data, x, y);
                    else
                        return trsv_coo_u_hi_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->col_data, (J*)A->mat->val_data, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_coo_n_lo_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->col_data, (J*)A->mat->val_data, x, y);
                    else
                        return trsv_coo_u_lo_plain(alpha, A->mat->rows, A->mat->cols, A->mat->nnz, A->mat->row_data, A->mat->col_data, (J*)A->mat->val_data, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_coo_n_hi_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_coo_u_hi_trans_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_coo_n_lo_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_coo_u_lo_trans_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_coo_n_hi_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_coo_u_hi_conj_plain(alpha, A->mat, x, y);
                }else if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_coo_n_lo_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_coo_u_lo_conj_plain(alpha, A->mat, x, y);
                }
                else
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                return diagsv_coo_n_plain(alpha, A->mat, x, y);
            else
                return diagsv_coo_u_plain(alpha, A->mat, x, y);
        }
        else
        {
            fprintf(stderr,"matrix type not supported\n");
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }  
    }    
    else if(A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        //TRIANG
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {   
            //not trans
            if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csc_n_hi_plain(alpha, A->mat, x, y);
                    else
                        return trsv_csc_u_hi_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csc_n_lo_plain(alpha, A->mat, x, y);
                    else
                        return trsv_csc_u_lo_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csc_n_hi_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_csc_u_hi_trans_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csc_n_lo_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_csc_u_lo_trans_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csc_n_hi_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_csc_u_hi_conj_plain(alpha, A->mat, x, y);
                }else if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_csc_n_lo_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_csc_u_lo_conj_plain(alpha, A->mat, x, y);
                }
                else
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                return diagsv_csc_n_plain(alpha, A->mat, x, y);
            else
                return diagsv_csc_u_plain(alpha, A->mat, x, y);
        }
        else
        {
            fprintf(stderr,"matrix type not supported\n");
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }  
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        //TRIANG
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {   
            //not trans
            if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_bsr_n_hi_plain(alpha, A->mat, x, y);
                    else
                        return trsv_bsr_u_hi_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_bsr_n_lo_plain(alpha, A->mat, x, y);
                    else
                        return trsv_bsr_u_lo_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_bsr_n_hi_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_bsr_u_hi_trans_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_bsr_n_lo_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_bsr_u_lo_trans_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_bsr_n_hi_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_bsr_u_hi_conj_plain(alpha, A->mat, x, y);
                }else if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_bsr_n_lo_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_bsr_u_lo_conj_plain(alpha, A->mat, x, y);
                }
                else
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                return diagsv_bsr_n_plain(alpha, A->mat, x, y);
            else
                return diagsv_bsr_u_plain(alpha, A->mat, x, y);
        }
        else
        {
            fprintf(stderr,"matrix type not supported\n");
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }  
    }
        else if(A->format == ALPHA_SPARSE_FORMAT_DIA)
    {
        //TRIANG
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {   
            //not trans
            if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_dia_n_hi_plain(alpha, A->mat, x, y);
                    else
                        return trsv_dia_u_hi_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_dia_n_lo_plain(alpha, A->mat, x, y);
                    else
                        return trsv_dia_u_lo_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_dia_n_hi_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_dia_u_hi_trans_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_dia_n_lo_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_dia_u_lo_trans_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_dia_n_hi_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_dia_u_hi_conj_plain(alpha, A->mat, x, y);
                }else if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_dia_n_lo_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_dia_u_lo_conj_plain(alpha, A->mat, x, y);
                }
                else
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                return diagsv_dia_n_plain(alpha, A->mat, x, y);
            else
                return diagsv_dia_u_plain(alpha, A->mat, x, y);
        }
        else
        {
            fprintf(stderr,"matrix type not supported\n");
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }  
    }
        else if(A->format == ALPHA_SPARSE_FORMAT_SKY)
    {
        //TRIANG
        if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR)
        {   
            //not trans
            if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_sky_n_hi_plain(alpha, A->mat, x, y);
                    else
                        return trsv_sky_u_hi_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_sky_n_lo_plain(alpha, A->mat, x, y);
                    else
                        return trsv_sky_u_lo_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_sky_n_hi_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_sky_u_hi_trans_plain(alpha, A->mat, x, y);
                }
                else{
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_sky_n_lo_trans_plain(alpha, A->mat, x, y);
                    else
                        return trsv_sky_u_lo_trans_plain(alpha, A->mat, x, y);
                }
            }
            else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE){
                if(descr.mode == ALPHA_SPARSE_FILL_MODE_UPPER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_sky_n_hi_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_sky_u_hi_conj_plain(alpha, A->mat, x, y);
                }else if(descr.mode == ALPHA_SPARSE_FILL_MODE_LOWER){
                    if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                        return trsv_sky_n_lo_conj_plain(alpha, A->mat, x, y);
                    else
                        return trsv_sky_u_lo_conj_plain(alpha, A->mat, x, y);
                }
                else
                    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
            }
            else
                return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }
        else if (descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL)
        {
            if(descr.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
                return diagsv_sky_n_plain(alpha, A->mat, x, y);
            else
                return diagsv_sky_u_plain(alpha, A->mat, x, y);
        }
        else
        {
            fprintf(stderr,"matrix type not supported\n");
            return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
        }  
    }
    else
    {
        fprintf(stderr,"format not supported\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }  
}


#define C_IMPL(ONAME, TYPE)                                               \
    alphasparseStatus_t ONAME(const alphasparseOperation_t operation,     \
                            const TYPE alpha,                             \
                            const alphasparse_matrix_t A,                 \
                            const struct alpha_matrix_descr descr,        \
                            const TYPE *x,                                \
                            TYPE *y)                                      \
    {                                                                     \
        return alphasparse_trsv_template_plain(operation,              \
                            alpha,                                     \
                            A,                                         \
                            descr,                                     \
                            x,                                         \
                            y);                                        \
    }                                                                  \
                                             
C_IMPL(alphasparse_s_trsv_plain, float);
C_IMPL(alphasparse_d_trsv_plain, double);
C_IMPL(alphasparse_c_trsv_plain, ALPHA_Complex8);
C_IMPL(alphasparse_z_trsv_plain, ALPHA_Complex16);
#undef C_IMPL