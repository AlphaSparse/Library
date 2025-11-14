#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
#include "alphasparse/util.h"
#include "spmmd/spmmd.h"
#include <cstdio>

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t alphasparse_spmmd_template(const alphasparseOperation_t operation, const alphasparse_matrix_t A,
                          const alphasparse_matrix_t B,
                          const alphasparse_layout_t layout, /* storage scheme for the output dense
                                                               matrix: C-style or Fortran-style */
                          J *matC, const ALPHA_INT ldc) {
  check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(B->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(matC, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

  check_return(A->format != B->format, ALPHA_SPARSE_STATUS_INVALID_VALUE);
  // check_return(A->datatype_cpu != ALPHA_SPARSE_DATATYPE_FLOAT, ALPHA_SPARSE_STATUS_INVALID_VALUE);
  // check_return(B->datatype_cpu != ALPHA_SPARSE_DATATYPE_FLOAT, ALPHA_SPARSE_STATUS_INVALID_VALUE);

  if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  // check if colA == rowB
  check_return(B->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);

  if (A->format == ALPHA_SPARSE_FORMAT_CSR) {
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
      if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return spmmd_csr_row<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return spmmd_csr_row_trans<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return spmmd_csr_col_trans<J>(A->mat, B->mat, matC, ldc);
      else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
      if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return spmmd_csr_col<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return spmmd_csr_col_trans<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return spmmd_csr_col_conj<J>(A->mat, B->mat, matC, ldc);
      else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    else 
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } 
  else if (A->format == ALPHA_SPARSE_FORMAT_CSC) {
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
      if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return spmmd_csc_row<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return spmmd_csc_row_trans<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return spmmd_csc_col_trans<J>(A->mat, B->mat, matC, ldc);
      else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
      if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return spmmd_csc_col<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return spmmd_csc_col_trans<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return spmmd_csc_col_conj<J>(A->mat, B->mat, matC, ldc);
      else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    else 
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else if (A->format == ALPHA_SPARSE_FORMAT_BSR) {
    if(layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
    {
      if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return spmmd_bsr_row<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return spmmd_bsr_row_trans<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return spmmd_bsr_col_trans<J>(A->mat, B->mat, matC, ldc);
      else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    else if(layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
    {
      if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return spmmd_bsr_col<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return spmmd_bsr_col_trans<J>(A->mat, B->mat, matC, ldc);
      else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return spmmd_bsr_col_conj<J>(A->mat, B->mat, matC, ldc);
      else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
    else 
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  else
      return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#define C_IMPL(ONAME, TYPE)                                                                             \
    alphasparseStatus_t ONAME(const alphasparseOperation_t operation, const alphasparse_matrix_t A,     \
                              const alphasparse_matrix_t B,                                             \
                              const alphasparse_layout_t layout, /*storage scheme for the output dense*/\
                              TYPE *matC, const ALPHA_INT ldc) {                                        \
        return alphasparse_spmmd_template(operation, A, B, layout, matC, ldc);                          \
    }
C_IMPL(alphasparse_s_spmmd, float);
C_IMPL(alphasparse_d_spmmd, double);
C_IMPL(alphasparse_c_spmmd, ALPHA_Complex8);
C_IMPL(alphasparse_z_spmmd, ALPHA_Complex16);
#undef C_IMPL