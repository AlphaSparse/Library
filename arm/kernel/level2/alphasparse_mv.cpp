/**
 * @brief implement for alphasparse_?_mv intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/inspector.h"
#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
// #include "alphasparse/tuning.h"
#include "alphasparse/util.h"
#include "./mv/gemv/gemv.h"
#include "./mv/hermv/hermv.h"
#include "./mv/diagmv/diagmv.h"
#include "./mv/symv/symv.h"
#include "./mv/trmv/trmv.h"
#include <cstdio>

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t alphasparse_mv_template(  
    const alphasparseOperation_t op_rq,  // operation_request
    const J alpha, 
    const alphasparse_matrix_t A,
    const struct alpha_matrix_descr dscr_rq, 
    const J *x, const J beta, J *y) {
  check_null_return(A, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(x, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(y, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  
  check_return(!((A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)||
               (A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)||
               (A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)||
               (A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)),
               ALPHA_SPARSE_STATUS_INVALID_VALUE);
  // TODO use simplelist to record optimized history
  // alphasparse_matrix_t compute_mat = NULL;
  struct alpha_matrix_descr compute_descr = dscr_rq;
  alphasparseOperation_t compute_operation = op_rq;

  if (dscr_rq.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC ||
      dscr_rq.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN)
    // check if it is a square matrix
    check_return(A->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);

  void * mat = A->mat;
  if (A->format == ALPHA_SPARSE_FORMAT_CSR) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_csr(alpha, A->mat->rows, A->mat->cols, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data, (J*)(A->mat->val_data), x, beta, y);
      else if(op_rq == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return gemv_csr_trans(alpha, A->mat->rows, A->mat->cols, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, beta, y);
      else
        return gemv_csr_conj(alpha, A->mat->rows, A->mat->cols, A->mat->row_data, A->mat->row_data + 1, A->mat->col_data,  (J*)(A->mat->val_data), x, beta, y);
    } else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csr_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csr_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csr_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csr_u_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csr_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csr_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csr_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csr_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csr_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csr_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csr_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csr_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csr_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csr_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csr_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csr_u_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csr_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csr_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csr_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csr_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csr_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csr_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csr_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csr_u_hi_trans<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csr_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csr_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csr_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csr_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
        return diagmv_csr_n<J>(alpha, A->mat, x, beta, y);
      else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
        return diagmv_csr_u<J>(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    } else {
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
  }
  else if (A->format == ALPHA_SPARSE_FORMAT_COO) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_coo(alpha, A->mat, x, beta, y);
      else if(op_rq == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return gemv_coo_trans(alpha, A->mat, x, beta, y);
      else
        return gemv_coo_conj(alpha, A->mat, x, beta, y);
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
        return diagmv_coo_n<J>(alpha, A->mat, x, beta, y);
      else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
        return diagmv_coo_u<J>(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_coo_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_coo_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_coo_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_coo_u_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_coo_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_coo_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_coo_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_coo_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_coo_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_coo_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_coo_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_coo_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_coo_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_coo_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_coo_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_coo_u_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_coo_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_coo_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_coo_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_coo_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_coo_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_coo_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_coo_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_coo_u_hi_trans<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_coo_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_coo_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_coo_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_coo_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } 
  else if (A->format == ALPHA_SPARSE_FORMAT_CSC) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_csc(alpha, A->mat, x, beta, y);
      else if(op_rq == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return gemv_csc_trans(alpha, A->mat, x, beta, y);
      else
        return gemv_csc_conj(alpha, A->mat, x, beta, y);
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
        return diagmv_csc_n<J>(alpha, A->mat, x, beta, y);
      else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
        return diagmv_csc_u<J>(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csc_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csc_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csc_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csc_u_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csc_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csc_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_csc_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_csc_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csc_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csc_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csc_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csc_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csc_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csc_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_csc_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_csc_u_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csc_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csc_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csc_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csc_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csc_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csc_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csc_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csc_u_hi_trans<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csc_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csc_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_csc_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_csc_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else if (A->format == ALPHA_SPARSE_FORMAT_BSR) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_bsr(alpha, A->mat, x, beta, y);
      else if(op_rq == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return gemv_bsr_trans(alpha, A->mat, x, beta, y);
      else
        return gemv_bsr_conj(alpha, A->mat, x, beta, y);
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
        return diagmv_bsr_n<J>(alpha, A->mat, x, beta, y);
      else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
        return diagmv_bsr_u<J>(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_bsr_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_bsr_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_bsr_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_bsr_u_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_bsr_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_bsr_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_bsr_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_bsr_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_bsr_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_bsr_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_bsr_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_bsr_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_bsr_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_bsr_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_bsr_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_bsr_u_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_bsr_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_bsr_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_bsr_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_bsr_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_bsr_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_bsr_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_bsr_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_bsr_u_hi_trans<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_bsr_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_bsr_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_bsr_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_bsr_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else if (A->format == ALPHA_SPARSE_FORMAT_DIA) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_dia(alpha, A->mat, x, beta, y);
      else if(op_rq == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return gemv_dia_trans(alpha, A->mat, x, beta, y);
      else
        return gemv_dia_conj(alpha, A->mat, x, beta, y);
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
        return diagmv_dia_n<J>(alpha, A->mat, x, beta, y);
      else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
        return diagmv_dia_u<J>(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_dia_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_dia_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_dia_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_dia_u_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_dia_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_dia_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_dia_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_dia_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_dia_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_dia_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_dia_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_dia_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_dia_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_dia_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_dia_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_dia_u_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_dia_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_dia_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_dia_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_dia_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_dia_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_dia_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_dia_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_dia_u_hi_trans<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_dia_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_dia_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_dia_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_dia_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else if (A->format == ALPHA_SPARSE_FORMAT_ELL) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_GENERAL) {
      if (op_rq == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) 
        return gemv_ell(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else if (A->format == ALPHA_SPARSE_FORMAT_SKY) { 
    if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL) {
      if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
        return diagmv_sky_n<J>(alpha, A->mat, x, beta, y);
      else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
        return diagmv_sky_u<J>(alpha, A->mat, x, beta, y);
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC) {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_sky_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_sky_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_sky_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_sky_u_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      if(compute_operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_sky_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_sky_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return symv_sky_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return symv_sky_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_sky_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_sky_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_sky_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_sky_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_sky_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_sky_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return hermv_sky_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return hermv_sky_u_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (compute_descr.type == ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR) 
    {
      if(compute_operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
      {
          if(dscr_rq.diag == ALPHA_SPARSE_DIAG_NON_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_sky_n_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_sky_n_hi<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_sky_u_lo<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_sky_u_hi<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_sky_n_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_sky_n_hi_trans<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_sky_u_lo_trans<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_sky_u_hi_trans<J>(alpha, A->mat, x, beta, y);
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
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_sky_n_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_sky_n_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else if(dscr_rq.diag == ALPHA_SPARSE_DIAG_UNIT)
          {
            if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_LOWER)
                return trmv_sky_u_lo_conj<J>(alpha, A->mat, x, beta, y);
            else if(dscr_rq.mode == ALPHA_SPARSE_FILL_MODE_UPPER)
                return trmv_sky_u_hi_conj<J>(alpha, A->mat, x, beta, y);
            else
                return ALPHA_SPARSE_STATUS_INVALID_VALUE;
          }
          else
              return ALPHA_SPARSE_STATUS_INVALID_VALUE;
      }
      else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else {
    fprintf(stderr, "format not supported\n");
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
}

#define C_IMPL(ONAME, TYPE)                                                          \
    alphasparseStatus_t ONAME(                                                       \
        const alphasparseOperation_t op_rq,  /*operation_request*/                   \
        const TYPE alpha, const alphasparse_matrix_t A,                              \
        const struct alpha_matrix_descr  dscr_rq,                                    \
        /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +  alphasparse_diag_type_t */ \
        const TYPE *x, const TYPE beta, TYPE *y)                                     \
    {                                                                                \
        return alphasparse_mv_template(op_rq,                                        \
                                       alpha,                                        \
                                       A,                                            \
                                       dscr_rq,                                      \
                                       x,                                            \
                                       beta,                                         \
                                       y);                                           \
    }

C_IMPL(alphasparse_s_mv, float);
C_IMPL(alphasparse_d_mv, double);
C_IMPL(alphasparse_c_mv, ALPHA_Complex8);
C_IMPL(alphasparse_z_mv, ALPHA_Complex16);
#undef C_IMPL