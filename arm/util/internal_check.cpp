#include <assert.h>
#include <math.h>
#include <memory.h>

#include "alphasparse/spapi.h"  // spblas API
#include "alphasparse/util.h"
bool check_equal_row_col(const alphasparse_matrix_t A) {
  if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
      return ((spmat_csr_s_t *)A->mat)->rows == ((spmat_csr_s_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_COO)
      return ((spmat_coo_s_t *)A->mat)->rows == ((spmat_coo_s_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
      return ((spmat_csc_s_t *)A->mat)->rows == ((spmat_csc_s_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
      return ((spmat_bsr_s_t *)A->mat)->rows == ((spmat_bsr_s_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
      return ((spmat_dia_s_t *)A->mat)->rows == ((spmat_dia_s_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
      return ((spmat_sky_s_t *)A->mat)->rows == ((spmat_sky_s_t *)A->mat)->cols;
    else
      return false;
  } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
      return ((spmat_csr_d_t *)A->mat)->rows == ((spmat_csr_d_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_COO)
      return ((spmat_coo_d_t *)A->mat)->rows == ((spmat_coo_d_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
      return ((spmat_csc_d_t *)A->mat)->rows == ((spmat_csc_d_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
      return ((spmat_bsr_d_t *)A->mat)->rows == ((spmat_bsr_d_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
      return ((spmat_dia_d_t *)A->mat)->rows == ((spmat_dia_d_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
      return ((spmat_sky_d_t *)A->mat)->rows == ((spmat_sky_d_t *)A->mat)->cols;
    else
      return false;
  } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
      return ((spmat_csr_c_t *)A->mat)->rows == ((spmat_csr_c_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_COO)
      return ((spmat_coo_c_t *)A->mat)->rows == ((spmat_coo_c_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
      return ((spmat_csc_c_t *)A->mat)->rows == ((spmat_csc_c_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
      return ((spmat_bsr_c_t *)A->mat)->rows == ((spmat_bsr_c_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
      return ((spmat_dia_c_t *)A->mat)->rows == ((spmat_dia_c_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
      return ((spmat_sky_c_t *)A->mat)->rows == ((spmat_sky_c_t *)A->mat)->cols;
    else
      return false;
  } else {
    if (A->format == ALPHA_SPARSE_FORMAT_CSR)
      return ((spmat_csr_z_t *)A->mat)->rows == ((spmat_csr_z_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_COO)
      return ((spmat_coo_z_t *)A->mat)->rows == ((spmat_coo_z_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
      return ((spmat_csc_z_t *)A->mat)->rows == ((spmat_csc_z_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
      return ((spmat_bsr_z_t *)A->mat)->rows == ((spmat_bsr_z_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
      return ((spmat_dia_z_t *)A->mat)->rows == ((spmat_dia_z_t *)A->mat)->cols;
    else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
      return ((spmat_sky_z_t *)A->mat)->rows == ((spmat_sky_z_t *)A->mat)->cols;
    else
      return false;
  }
}

bool check_equal_colA_rowB(const alphasparse_matrix_t A, const alphasparse_matrix_t B,
                           const alphasparseOperation_t transA) {
  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_s_t *)B->mat)->rows == ((spmat_csr_s_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_s_t *)B->mat)->rows == ((spmat_coo_s_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_s_t *)B->mat)->rows == ((spmat_csc_s_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_s_t *)B->mat)->rows == ((spmat_bsr_s_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_s_t *)B->mat)->rows == ((spmat_dia_s_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_s_t *)B->mat)->rows == ((spmat_sky_s_t *)A->mat)->cols;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_d_t *)B->mat)->rows == ((spmat_csr_d_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_d_t *)B->mat)->rows == ((spmat_coo_d_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_d_t *)B->mat)->rows == ((spmat_csc_d_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_d_t *)B->mat)->rows == ((spmat_bsr_d_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_d_t *)B->mat)->rows == ((spmat_dia_d_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_d_t *)B->mat)->rows == ((spmat_sky_d_t *)A->mat)->cols;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_c_t *)B->mat)->rows == ((spmat_csr_c_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_c_t *)B->mat)->rows == ((spmat_coo_c_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_c_t *)B->mat)->rows == ((spmat_csc_c_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_c_t *)B->mat)->rows == ((spmat_bsr_c_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_c_t *)B->mat)->rows == ((spmat_dia_c_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_c_t *)B->mat)->rows == ((spmat_sky_c_t *)A->mat)->cols;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    } else {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_z_t *)B->mat)->rows == ((spmat_csr_z_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_z_t *)B->mat)->rows == ((spmat_coo_z_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_z_t *)B->mat)->rows == ((spmat_csc_z_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_z_t *)B->mat)->rows == ((spmat_bsr_z_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_z_t *)B->mat)->rows == ((spmat_dia_z_t *)A->mat)->cols;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_z_t *)B->mat)->rows == ((spmat_sky_z_t *)A->mat)->cols;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    }
  } else if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
             transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_s_t *)B->mat)->rows == ((spmat_csr_s_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_s_t *)B->mat)->rows == ((spmat_coo_s_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_s_t *)B->mat)->rows == ((spmat_csc_s_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_s_t *)B->mat)->rows == ((spmat_bsr_s_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_s_t *)B->mat)->rows == ((spmat_dia_s_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_s_t *)B->mat)->rows == ((spmat_sky_s_t *)A->mat)->rows;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    } else if (A->datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_d_t *)B->mat)->rows == ((spmat_csr_d_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_d_t *)B->mat)->rows == ((spmat_coo_d_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_d_t *)B->mat)->rows == ((spmat_csc_d_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_d_t *)B->mat)->rows == ((spmat_bsr_d_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_d_t *)B->mat)->rows == ((spmat_dia_d_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_d_t *)B->mat)->rows == ((spmat_sky_d_t *)A->mat)->rows;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    } else if (A->datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_c_t *)B->mat)->rows == ((spmat_csr_c_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_c_t *)B->mat)->rows == ((spmat_coo_c_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_c_t *)B->mat)->rows == ((spmat_csc_c_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_c_t *)B->mat)->rows == ((spmat_bsr_c_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_c_t *)B->mat)->rows == ((spmat_dia_c_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_c_t *)B->mat)->rows == ((spmat_sky_c_t *)A->mat)->rows;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    } else {
      if (A->format == ALPHA_SPARSE_FORMAT_CSR)
        return ((spmat_csr_z_t *)B->mat)->rows == ((spmat_csr_z_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_COO)
        return ((spmat_coo_z_t *)B->mat)->rows == ((spmat_coo_z_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_CSC)
        return ((spmat_csc_z_t *)B->mat)->rows == ((spmat_csc_z_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_BSR)
        return ((spmat_bsr_z_t *)B->mat)->rows == ((spmat_bsr_z_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_DIA)
        return ((spmat_dia_z_t *)B->mat)->rows == ((spmat_dia_z_t *)A->mat)->rows;
      else if (A->format == ALPHA_SPARSE_FORMAT_SKY)
        return ((spmat_sky_z_t *)B->mat)->rows == ((spmat_sky_z_t *)A->mat)->rows;
      else {
        printf("A->cols != B->rows");
        return false;
      }
    }
  } else {
    assert(0);
  }
}

bool check_data_type(alphasparseDataType dt)
{
  if (dt == ALPHA_R_32F || dt == ALPHA_R_64F || dt == ALPHA_C_32F || dt == ALPHA_C_64F || dt == ALPHA_R_16F || dt == ALPHA_C_16F )
  {
    return true;
  }
  else
    return false;  
}

bool check_data_type(alphasparse_datatype_t dt)
{
  if (dt == ALPHA_SPARSE_DATATYPE_FLOAT || dt == ALPHA_SPARSE_DATATYPE_DOUBLE || dt == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX || dt == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX || dt == ALPHA_SPARSE_DATATYPE_HALF_FLOAT || dt == ALPHA_SPARSE_DATATYPE_HALF_DOUBLE )
  {
    return true;
  }
  else
    return false;  
}