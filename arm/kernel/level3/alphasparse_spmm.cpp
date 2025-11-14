#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
#include "alphasparse/util.h"
#include "spmm/spmm.h"
#include <cstdio>

alphasparseStatus_t alphasparse_spmm(const alphasparseOperation_t operation,
                                    const alphasparse_matrix_t A, const alphasparse_matrix_t B,
                                    alphasparse_matrix_t *C) {
  check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(B->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(C, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

  check_return(A->datatype_cpu != B->datatype_cpu, ALPHA_SPARSE_STATUS_INVALID_VALUE);
  check_return(A->format != B->format, ALPHA_SPARSE_STATUS_INVALID_VALUE);

  // check if colA == rowB
  check_return(B->mat->rows != A->mat->cols, ALPHA_SPARSE_STATUS_INVALID_VALUE);

  if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;

  alphasparse_matrix *CC = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
  *C = CC;

  CC->datatype_cpu = A->datatype_cpu;
  CC->format = A->format;
  if (A->format == ALPHA_SPARSE_FORMAT_CSR) {
    if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_csr<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_csr<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_csr<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_csr<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }   
    else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)   
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_csr_trans<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_csr_trans<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_csr_trans<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_csr_trans<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)   
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_csr_conj<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_csr_conj<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_csr_conj<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_csr_conj<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else
      return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }   
  else if (A->format == ALPHA_SPARSE_FORMAT_BSR) {
    if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_bsr<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_bsr<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_bsr<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_bsr<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }   
    else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)   
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_bsr_trans<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_bsr_trans<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_bsr_trans<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_bsr_trans<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)   
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_bsr_conj<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_bsr_conj<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_bsr_conj<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_bsr_conj<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else
      return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }   
  else if (A->format == ALPHA_SPARSE_FORMAT_CSC) {
    if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_csc<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_csc<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_csc<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_csc<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }   
    else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)   
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_csc_trans<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_csc_trans<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_csc_trans<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_csc_trans<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)   
    {
      if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT)
        return spmm_csc_conj<float>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE)
        return spmm_csc_conj<double>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
        return spmm_csc_conj<ALPHA_Complex8>(A->mat, B->mat, &(CC->mat));
      else if(A->datatype_cpu == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
        return spmm_csc_conj<ALPHA_Complex16>(A->mat, B->mat, &(CC->mat));
      else
      {
        printf("shouldn't come here\n");
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
      }
    }
    else
      return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }                              
  printf("shouldn't come here\n");
  return ALPHA_SPARSE_STATUS_INVALID_VALUE;
}
