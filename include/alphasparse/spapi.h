#pragma once

#include "spdef.h"
#include "spmat.h"
#ifdef __CUDA__
#include <cuComplex.h>
#include <cuda_fp16.h>
#endif
#ifdef __HIP__
#include <hip/hip_complex.h>
#define cuFloatComplex hipFloatComplex
#define cuDoubleComplex hipDoubleComplex
#endif
#ifdef __HYGON__
#define cuFloatComplex ALPHA_Complex8
#define cuDoubleComplex ALPHA_Complex16
#endif
#ifdef __PLAIN__
#define cuFloatComplex ALPHA_Complex8
#define cuDoubleComplex ALPHA_Complex16
#endif
#ifdef __ARM__
#define cuFloatComplex ALPHA_Complex8
#define cuDoubleComplex ALPHA_Complex16
#endif
alphasparseStatus_t
alphasparse_transpose(const alphasparse_matrix_t source,
                      alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_csc(const alphasparse_matrix_t source,
                        const alphasparseOperation_t operation,
                        alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_bsr(
  const alphasparse_matrix_t
    source, /* convert original matrix to BSR representation */
  const int block_size,
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const alphasparseOperation_t
    operation, /* as is, transposed or conjugate transposed */
  alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_sky(const alphasparse_matrix_t source,
                        const alphasparseOperation_t operation,
                        const alphasparse_fill_mode_t fill,
                        alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_dia(const alphasparse_matrix_t source,
                        const alphasparseOperation_t operation,
                        alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_ell(const alphasparse_matrix_t source,
                        const alphasparseOperation_t operation,
                        alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_hyb(const alphasparse_matrix_t source,
                        const alphasparseOperation_t operation,
                        alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_csr5(const alphasparse_matrix_t source,
                         const alphasparseOperation_t operation,
                         alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_cooaos(const alphasparse_matrix_t source,
                           const alphasparseOperation_t operation,
                           alphasparse_matrix_t* dest);
alphasparseStatus_t
alphasparse_convert_sell_csigma(
  const alphasparse_matrix_t
    source, /* convert original matrix to SELL_C_Sgima representation */
  const bool SHORT_BINNING,
  const int C,
  const int SIGMA,
  const alphasparseOperation_t
    operation, /* as is, transposed or conjugate transposed */
  alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_ellr(const alphasparse_matrix_t source,
                         const alphasparseOperation_t operation,
                         alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_hints_bsr(
  const alphasparse_matrix_t
    source, /* convert original matrix to BSR representation */
  const int block_size,
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const alphasparseOperation_t
    operation, /* as is, transposed or conjugate transposed */
  alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_hints_dia(const alphasparse_matrix_t source,
                              const alphasparseOperation_t operation,
                              alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_hints_ell(const alphasparse_matrix_t source,
                              const alphasparseOperation_t operation,
                              alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_gebsr(
  const alphasparse_matrix_t
    source, /* convert original matrix to GEBSR representation */
  const int block_row_dim,
  const int block_col_dim,
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const alphasparseOperation_t
    operation, /* as is, transposed or conjugate transposed */
  alphasparse_matrix_t* dest);

template <typename T>
alphasparseStatus_t
alphasparseXcoo2csr(const int* row_data, int nnz, int m, T* csrRowPtr);
/**
 * --------------------------------------------------------------------------------------
 */

/*****************************************************************************************/
/*************************************** Creation routines
 * *******************************/
/*****************************************************************************************/

/*
    Matrix handle is used for storing information about the matrix and matrix
   values

    Create matrix from one of the existing sparse formats by creating the handle
   with matrix info and copy matrix values if requested. Collect high-level info
   about the matrix. Need to use this interface for the case with several calls
   in program for performance reasons, where optimizations are not required.

    coordinate format,
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are
   stored in the handle

    *** User data is not marked const since the alphasparse_order() or
   alphasparse_?_set_values() functionality could change user data.  However,
   this is only done by a user call. Internally const-ness of user data is
   maintained other than through explicit use of these interfaces.

*/

alphasparseStatus_t
alphasparse_s_create_coo(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         const int nnz,
                         int* row_indx,
                         int* col_indx,
                         float* values);

alphasparseStatus_t
alphasparse_d_create_coo(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         const int nnz,
                         int* row_indx,
                         int* col_indx,
                         double* values);

alphasparseStatus_t
alphasparse_c_create_coo(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         const int nnz,
                         int* row_indx,
                         int* col_indx,
                         cuFloatComplex* values);

alphasparseStatus_t
alphasparse_c_f32_create_coo(
  alphasparse_matrix_t* A,
  const alphasparseIndexBase_t
    indexing, /* indexing: C-style or Fortran-style */
  const int rows,
  const int cols,
  const int nnz,
  int* row_indx,
  int* col_indx,
  cuFloatComplex* values);

alphasparseStatus_t
alphasparse_z_create_coo(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         const int nnz,
                         int* row_indx,
                         int* col_indx,
                         cuDoubleComplex* values);

/*
    compressed sparse row format (4-arrays version),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are
   stored in the handle

    *** User data is not marked const since the alphasparse_order() or
   alphasparse_?_set_values() functionality could change user data.  However,
   this is only done by a user call. Internally const-ness of user data is
   maintained other than through explicit use of these interfaces.


*/
alphasparseStatus_t
alphasparse_s_create_csr(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* rows_start,
                         int* rows_end,
                         int* col_indx,
                         float* values);

alphasparseStatus_t
alphasparse_d_create_csr(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* rows_start,
                         int* rows_end,
                         int* col_indx,
                         double* values);

alphasparseStatus_t
alphasparse_c_create_csr(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* rows_start,
                         int* rows_end,
                         int* col_indx,
                         cuFloatComplex* values);

alphasparseStatus_t
alphasparse_z_create_csr(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* rows_start,
                         int* rows_end,
                         int* col_indx,
                         cuDoubleComplex* values);

/*
    compressed sparse column format (4-arrays version),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are
   stored in the handle

    *** User data is not marked const since the alphasparse_order() or
   alphasparse_?_set_values() functionality could change user data.  However,
   this is only done by a user call. Internally const-ness of user data is
   maintained other than through explicit use of these interfaces.

*/
alphasparseStatus_t
alphasparse_s_create_csc(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* cols_start,
                         int* cols_end,
                         int* row_indx,
                         float* values);

alphasparseStatus_t
alphasparse_d_create_csc(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* cols_start,
                         int* cols_end,
                         int* row_indx,
                         double* values);

alphasparseStatus_t
alphasparse_c_create_csc(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* cols_start,
                         int* cols_end,
                         int* row_indx,
                         cuFloatComplex* values);

alphasparseStatus_t
alphasparse_z_create_csc(alphasparse_matrix_t* A,
                         const alphasparseIndexBase_t
                           indexing, /* indexing: C-style or Fortran-style */
                         const int rows,
                         const int cols,
                         int* cols_start,
                         int* cols_end,
                         int* row_indx,
                         cuDoubleComplex* values);

/*
    compressed block sparse row format (4-arrays version, square blocks),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are
   stored in the handle

    *** User data is not marked const since the alphasparse_order() or
   alphasparse_?_set_values() functionality could change user data.  However,
   this is only done by a user call. Internally const-ness of user data is
   maintained other than through explicit use of these interfaces.

*/
alphasparseStatus_t
alphasparse_s_create_bsr(
  alphasparse_matrix_t* A,
  const alphasparseIndexBase_t
    indexing, /* indexing: C-style or Fortran-style */
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const int rows,
  const int cols,
  const int block_size,
  int* rows_start,
  int* rows_end,
  int* col_indx,
  float* values);

alphasparseStatus_t
alphasparse_d_create_bsr(
  alphasparse_matrix_t* A,
  const alphasparseIndexBase_t
    indexing, /* indexing: C-style or Fortran-style */
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const int rows,
  const int cols,
  const int block_size,
  int* rows_start,
  int* rows_end,
  int* col_indx,
  double* values);

alphasparseStatus_t
alphasparse_c_create_bsr(
  alphasparse_matrix_t* A,
  const alphasparseIndexBase_t
    indexing, /* indexing: C-style or Fortran-style */
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const int rows,
  const int cols,
  const int block_size,
  int* rows_start,
  int* rows_end,
  int* col_indx,
  cuFloatComplex* values);

alphasparseStatus_t
alphasparse_z_create_bsr(
  alphasparse_matrix_t* A,
  const alphasparseIndexBase_t
    indexing, /* indexing: C-style or Fortran-style */
  const alphasparse_layout_t
    block_layout, /* block storage: row-major or column-major */
  const int rows,
  const int cols,
  const int block_size,
  int* rows_start,
  int* rows_end,
  int* col_indx,
  cuDoubleComplex* values);

#include "alphasparse/format.h"
#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/internal_check.h"
#include "alphasparse/util/malloc.h"

void
alphasparse_convert_csr_mat_valuetype(alphasparse_matrix_t csr,
                                      alphasparseDataType target_valuetype);

alphasparseStatus_t
alphasparse_convert_csr_valuetype(alphasparse_matrix_t csr,
                                  alphasparseDataType target_valuetype);

/*
    Create copy of the existing handle; matrix properties could be changed.
    For example it could be used for extracting triangular or diagonal parts
   from existing matrix.
*/
alphasparseStatus_t
alphasparse_copy(const alphasparse_matrix_t source,
                 const struct alpha_matrix_descr
                   descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t
                             + alphasparse_diag_type_t */
                 alphasparse_matrix_t* dest);

/*
    destroy matrix handle; if sparse matrix was stored inside the handle it also
   deallocates the matrix It is user's responsibility not to delete the handle
   with the matrix, if this matrix is shared with other handles
*/
alphasparseStatus_t
alphasparse_destroy(alphasparse_matrix_t A);
/*
    return extended error information from last operation;
    e.g. info about wrong input parameter, memory sizes that couldn't be
   allocated
*/
alphasparseStatus_t
alphasparse_get_error_info(alphasparse_matrix_t A,
                           int* info); /* unsupported currently */

/*****************************************************************************************/
/************************ Converters of internal representation
 * *************************/
/*****************************************************************************************/

/* converters from current format to another */
alphasparseStatus_t
alphasparse_convert_csr(
  const alphasparse_matrix_t
    source, /* convert original matrix to CSR representation */
  const alphasparseOperation_t
    operation, /* as is, transposed or conjugate transposed */
  alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_convert_coo(
  const alphasparse_matrix_t
    source, /* convert original matrix to CSR representation */
  const alphasparseOperation_t
    operation, /* as is, transposed or conjugate transposed */
  alphasparse_matrix_t* dest);

alphasparseStatus_t
alphasparse_s_export_coo(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** row_indx,
  int** col_indx,
  float** values,
  int* nnz);

alphasparseStatus_t
alphasparse_d_export_coo(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** row_indx,
  int** col_indx,
  double** values,
  int* nnz);

alphasparseStatus_t
alphasparse_c_export_coo(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** row_indx,
  int** col_indx,
  cuFloatComplex** values,
  int* nnz);

alphasparseStatus_t
alphasparse_z_export_coo(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** row_indx,
  int** col_indx,
  cuDoubleComplex** values,
  int* nnz);

alphasparseStatus_t
alphasparse_s_export_bsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_size,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  float** values);

alphasparseStatus_t
alphasparse_d_export_bsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_size,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  double** values);

alphasparseStatus_t
alphasparse_c_export_bsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_size,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  cuFloatComplex** values);

alphasparseStatus_t
alphasparse_z_export_bsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_size,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  cuDoubleComplex** values);

alphasparseStatus_t
alphasparse_s_export_csr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  float** values);

alphasparseStatus_t
alphasparse_d_export_csr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  double** values);

alphasparseStatus_t
alphasparse_c_export_csr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  cuFloatComplex** values);

alphasparseStatus_t
alphasparse_z_export_csr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  cuDoubleComplex** values);

alphasparseStatus_t
alphasparse_s_export_csc(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** cols_start,
  int** cols_end,
  int** row_indx,
  float** values);

alphasparseStatus_t
alphasparse_d_export_csc(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** cols_start,
  int** cols_end,
  int** row_indx,
  double** values);

alphasparseStatus_t
alphasparse_c_export_csc(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** cols_start,
  int** cols_end,
  int** row_indx,
  cuFloatComplex** values);

alphasparseStatus_t
alphasparse_z_export_csc(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int** cols_start,
  int** cols_end,
  int** row_indx,
  cuDoubleComplex** values);

alphasparseStatus_t
alphasparse_s_export_ell(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* width,
  int** col_indx,
  float** values);

alphasparseStatus_t
alphasparse_d_export_ell(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* width,
  int** col_indx,
  double** values);

alphasparseStatus_t
alphasparse_c_export_ell(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* width,
  int** col_indx,
  cuFloatComplex** values);

alphasparseStatus_t
alphasparse_z_export_ell(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* width,
  int** col_indx,
  cuDoubleComplex** values);

alphasparseStatus_t
alphasparse_s_export_gebsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_row_dim,
  int* block_col_dim,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  float** values);

alphasparseStatus_t
alphasparse_d_export_gebsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_row_dim,
  int* block_col_dim,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  double** values);

alphasparseStatus_t
alphasparse_c_export_gebsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_row_dim,
  int* block_col_dim,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  cuFloatComplex** values);

alphasparseStatus_t
alphasparse_z_export_gebsr(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  alphasparse_layout_t*
    block_layout, /* block storage: row-major or column-major */
  int* rows,
  int* cols,
  int* block_row_dim,
  int* block_col_dim,
  int** rows_start,
  int** rows_end,
  int** col_indx,
  cuDoubleComplex** values);

alphasparseStatus_t
alphasparse_s_export_hyb(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* nnz,
  int* ell_width,
  float** ell_val,
  int** ell_col_ind,
  float** coo_val,
  int** coo_row_val,
  int** coo_col_val);

alphasparseStatus_t
alphasparse_d_export_hyb(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* nnz,
  int* ell_width,
  double** ell_val,
  int** ell_col_ind,
  double** coo_val,
  int** coo_row_val,
  int** coo_col_val);

alphasparseStatus_t
alphasparse_c_export_hyb(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* nnz,
  int* ell_width,
  cuFloatComplex** ell_val,
  int** ell_col_ind,
  cuFloatComplex** coo_val,
  int** coo_row_val,
  int** coo_col_val);

alphasparseStatus_t
alphasparse_z_export_hyb(
  const alphasparse_matrix_t source,
  alphasparseIndexBase_t* indexing, /* indexing: C-style or Fortran-style */
  int* rows,
  int* cols,
  int* nnz,
  int* ell_width,
  cuDoubleComplex** ell_val,
  int** ell_col_ind,
  cuDoubleComplex** coo_val,
  int** coo_row_val,
  int** coo_col_val);

/*****************************************************************************************/
/************************** Step-by-step modification routines
 * ***************************/
/*****************************************************************************************/

/* update existing value in the matrix ( for internal storage only, should not
 * work with user-allocated matrices) */
alphasparseStatus_t
alphasparse_s_set_value(alphasparse_matrix_t A,
                        const int row,
                        const int col,
                        const float value);

alphasparseStatus_t
alphasparse_d_set_value(alphasparse_matrix_t A,
                        const int row,
                        const int col,
                        const double value);

alphasparseStatus_t
alphasparse_c_set_value(alphasparse_matrix_t A,
                        const int row,
                        const int col,
                        const cuFloatComplex value);

alphasparseStatus_t
alphasparse_z_set_value(alphasparse_matrix_t A,
                        const int row,
                        const int col,
                        const cuDoubleComplex value);

/* update existing values in the matrix for internal storage only
       can be used to either update all or selected values */
alphasparseStatus_t
alphasparse_s_update_values(alphasparse_matrix_t A,
                            const int nvalues,
                            const int* indx,
                            const int* indy,
                            float* values);

alphasparseStatus_t
alphasparse_d_update_values(alphasparse_matrix_t A,
                            const int nvalues,
                            const int* indx,
                            const int* indy,
                            double* values);

alphasparseStatus_t
alphasparse_c_update_values(alphasparse_matrix_t A,
                            const int nvalues,
                            const int* indx,
                            const int* indy,
                            cuFloatComplex* values);

alphasparseStatus_t
alphasparse_z_update_values(alphasparse_matrix_t A,
                            const int nvalues,
                            const int* indx,
                            const int* indy,
                            cuDoubleComplex* values);

/*****************************************************************************************/
/****************************** Verbose mode routine
 * *************************************/
/*****************************************************************************************/

/* allow to switch on/off verbose mode */
alphasparseStatus_t
alphasparse_set_verbose_mode(
  alpha_verbose_mode_t verbose); /* unsupported currently */

/*****************************************************************************************/
/****************************** Optimization routines
 * ************************************/
/*****************************************************************************************/

/* Describe expected operations with amount of iterations */
alphasparseStatus_t
alphasparse_set_mv_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t
    operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for
                  infinite amount of calls */
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const int expected_calls);

alphasparseStatus_t
alphasparse_set_dotmv_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t
    operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for
                  infinite amount of calls */
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const int expectedCalls);

alphasparseStatus_t
alphasparse_set_mmd_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t operation,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const int dense_matrix_size, /* amount of columns in dense matrix */
  const int expected_calls);

alphasparseStatus_t
alphasparse_set_sv_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t
    operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for
                  infinite amount of calls */
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const int expected_calls);

alphasparseStatus_t
alphasparse_set_sm_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t operation,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const int dense_matrix_size, /* amount of columns in dense matrix */
  const int expected_calls);

alphasparseStatus_t
alphasparse_set_mm_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t transA,
  const struct alpha_matrix_descr
    descrA, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
               alphasparse_diag_type_t */
  const alphasparse_matrix_t B,
  const alphasparseOperation_t transB,
  const struct alpha_matrix_descr descrB,
  const int expected_calls);

alphasparseStatus_t
alphasparse_set_symgs_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t
    operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for
                  infinite amount of calls */
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const int expected_calls);

alphasparseStatus_t
alphasparse_set_lu_smoother_hint(
  const alphasparse_matrix_t A,
  const alphasparseOperation_t operation,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const int expectedCalls);

/* Describe memory usage model */
alphasparseStatus_t
alphasparse_set_memory_hint(
  const alphasparse_matrix_t A,
  const alphasparse_memory_usage_t
    policy); /* ALPHA_SPARSE_MEMORY_AGGRESSIVE is default value */

/*
    Optimize matrix described by the handle. It uses hints (optimization and
   memory) that should be set up before this call. If hints were not explicitly
   defined, default vales are: ALPHA_SPARSE_OPERATION_NON_TRANSPOSE for
   matrix-vector multiply with infinite number of expected iterations.
*/
alphasparseStatus_t
alphasparse_optimize(alphasparse_matrix_t A);

/*****************************************************************************************/
/****************************** Computational routines
 * ***********************************/
/*****************************************************************************************/

alphasparseStatus_t
alphasparse_order(const alphasparse_matrix_t A);

/*
    Perform computations based on created matrix handle

    Level 2
*/
/*   Computes y = alpha * A * x + beta * y   */
alphasparseStatus_t
alphasparse_s_mv(const alphasparseOperation_t operation,
                 const float alpha,
                 const alphasparse_matrix_t A,
                 const struct alpha_matrix_descr
                   descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t
                             + alphasparse_diag_type_t */
                 const float* x,
                 const float beta,
                 float* y);

alphasparseStatus_t
alphasparse_d_mv(const alphasparseOperation_t operation,
                 const double alpha,
                 const alphasparse_matrix_t A,
                 const struct alpha_matrix_descr
                   descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t
                             + alphasparse_diag_type_t */
                 const double* x,
                 const double beta,
                 double* y);

alphasparseStatus_t
alphasparse_c_mv(const alphasparseOperation_t operation,
                 const cuFloatComplex alpha,
                 const alphasparse_matrix_t A,
                 const struct alpha_matrix_descr
                   descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t
                             + alphasparse_diag_type_t */
                 const cuFloatComplex* x,
                 const cuFloatComplex beta,
                 cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_mv(const alphasparseOperation_t operation,
                 const cuDoubleComplex alpha,
                 const alphasparse_matrix_t A,
                 const struct alpha_matrix_descr
                   descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t
                             + alphasparse_diag_type_t */
                 const cuDoubleComplex* x,
                 const cuDoubleComplex beta,
                 cuDoubleComplex* y);

/*    Computes y = alpha * A * x + beta * y  and d = <x, y> , the l2 inner
 * product */
alphasparseStatus_t
alphasparse_s_dotmv(
  const alphasparseOperation_t transA,
  const float alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const float* x,
  const float beta,
  float* y,
  float* d);

alphasparseStatus_t
alphasparse_d_dotmv(
  const alphasparseOperation_t transA,
  const double alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const double* x,
  const double beta,
  double* y,
  double* d);

alphasparseStatus_t
alphasparse_c_dotmv(
  const alphasparseOperation_t transA,
  const cuFloatComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const cuFloatComplex* x,
  const cuFloatComplex beta,
  cuFloatComplex* y,
  cuFloatComplex* d);

alphasparseStatus_t
alphasparse_z_dotmv(
  const alphasparseOperation_t transA,
  const cuDoubleComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const cuDoubleComplex* x,
  const cuDoubleComplex beta,
  cuDoubleComplex* y,
  cuDoubleComplex* d);

/*   Solves triangular system y = alpha * A^{-1} * x   */
alphasparseStatus_t
alphasparse_s_trsv(
  const alphasparseOperation_t operation,
  const float alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const float* x,
  float* y);

alphasparseStatus_t
alphasparse_d_trsv(
  const alphasparseOperation_t operation,
  const double alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const double* x,
  double* y);

alphasparseStatus_t
alphasparse_c_trsv(
  const alphasparseOperation_t operation,
  const cuFloatComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const cuFloatComplex* x,
  cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_trsv(
  const alphasparseOperation_t operation,
  const cuDoubleComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const cuDoubleComplex* x,
  cuDoubleComplex* y);

/*   Applies symmetric Gauss-Seidel preconditioner to symmetric system A * x =
 * b, */
/*   that is, it solves: */
/*      x0       = alpha*x */
/*      (L+D)*x1 = b - U*x0 */
/*      (D+U)*x  = b - L*x1 */
/*                                                                                */
/*   SYMGS_MV also returns y = A*x */
alphasparseStatus_t
alphasparse_s_symgs(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const struct alpha_matrix_descr descr,
                    const float alpha,
                    const float* b,
                    float* x);

alphasparseStatus_t
alphasparse_d_symgs(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const struct alpha_matrix_descr descr,
                    const double alpha,
                    const double* b,
                    double* x);

alphasparseStatus_t
alphasparse_c_symgs(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const struct alpha_matrix_descr descr,
                    const cuFloatComplex alpha,
                    const cuFloatComplex* b,
                    cuFloatComplex* x);

alphasparseStatus_t
alphasparse_z_symgs(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const struct alpha_matrix_descr descr,
                    const cuDoubleComplex alpha,
                    const cuDoubleComplex* b,
                    cuDoubleComplex* x);

alphasparseStatus_t
alphasparse_s_symgs_mv(const alphasparseOperation_t op,
                       const alphasparse_matrix_t A,
                       const struct alpha_matrix_descr descr,
                       const float alpha,
                       const float* b,
                       float* x,
                       float* y);

alphasparseStatus_t
alphasparse_d_symgs_mv(const alphasparseOperation_t op,
                       const alphasparse_matrix_t A,
                       const struct alpha_matrix_descr descr,
                       const double alpha,
                       const double* b,
                       double* x,
                       double* y);

alphasparseStatus_t
alphasparse_c_symgs_mv(const alphasparseOperation_t op,
                       const alphasparse_matrix_t A,
                       const struct alpha_matrix_descr descr,
                       const cuFloatComplex alpha,
                       const cuFloatComplex* b,
                       cuFloatComplex* x,
                       cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_symgs_mv(const alphasparseOperation_t op,
                       const alphasparse_matrix_t A,
                       const struct alpha_matrix_descr descr,
                       const cuDoubleComplex alpha,
                       const cuDoubleComplex* b,
                       cuDoubleComplex* x,
                       cuDoubleComplex* y);

/*   Computes an action of a preconditioner
         which corresponds to the approximate matrix decomposition A ≈
   (L+D)*E*(U+D) for the system Ax = b.

         L is lower triangular part of A
         U is upper triangular part of A
         D is diagonal values of A
         E is approximate diagonal inverse

         That is, it solves:
             r = rhs - A*x0
             (L + D)*E*(U + D)*dx = r
             x1 = x0 + dx                                        */

alphasparseStatus_t
alphasparse_s_lu_smoother(const alphasparseOperation_t op,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr,
                          const float* diag,
                          const float* approx_diag_inverse,
                          float* x,
                          const float* rhs);

alphasparseStatus_t
alphasparse_d_lu_smoother(const alphasparseOperation_t op,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr,
                          const double* diag,
                          const double* approx_diag_inverse,
                          double* x,
                          const double* rhs);

alphasparseStatus_t
alphasparse_c_lu_smoother(const alphasparseOperation_t op,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr,
                          const cuFloatComplex* diag,
                          const cuFloatComplex* approx_diag_inverse,
                          cuFloatComplex* x,
                          const cuFloatComplex* rhs);

alphasparseStatus_t
alphasparse_z_lu_smoother(const alphasparseOperation_t op,
                          const alphasparse_matrix_t A,
                          const struct alpha_matrix_descr descr,
                          const cuDoubleComplex* diag,
                          const cuDoubleComplex* approx_diag_inverse,
                          cuDoubleComplex* x,
                          const cuDoubleComplex* rhs);

/* Level 3 */

/*   Computes y = alpha * A * x + beta * y   */
alphasparseStatus_t
alphasparse_s_mm(
  const alphasparseOperation_t operation,
  const float alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const float* x,
  const int columns,
  const int ldx,
  const float beta,
  float* y,
  const int ldy);

alphasparseStatus_t
alphasparse_d_mm(
  const alphasparseOperation_t operation,
  const double alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const double* x,
  const int columns,
  const int ldx,
  const double beta,
  double* y,
  const int ldy);

alphasparseStatus_t
alphasparse_c_mm(
  const alphasparseOperation_t operation,
  const cuFloatComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const cuFloatComplex* x,
  const int columns,
  const int ldx,
  const cuFloatComplex beta,
  cuFloatComplex* y,
  const int ldy);

alphasparseStatus_t
alphasparse_z_mm(
  const alphasparseOperation_t operation,
  const cuDoubleComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const cuDoubleComplex* x,
  const int columns,
  const int ldx,
  const cuDoubleComplex beta,
  cuDoubleComplex* y,
  const int ldy);

/*   Solves triangular system y = alpha * A^{-1} * x   */
alphasparseStatus_t
alphasparse_s_trsm(
  const alphasparseOperation_t operation,
  const float alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const float* x,
  const int columns,
  const int ldx,
  float* y,
  const int ldy);

alphasparseStatus_t
alphasparse_d_trsm(
  const alphasparseOperation_t operation,
  const double alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const double* x,
  const int columns,
  const int ldx,
  double* y,
  const int ldy);

alphasparseStatus_t
alphasparse_c_trsm(
  const alphasparseOperation_t operation,
  const cuFloatComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const cuFloatComplex* x,
  const int columns,
  const int ldx,
  cuFloatComplex* y,
  const int ldy);

alphasparseStatus_t
alphasparse_z_trsm(
  const alphasparseOperation_t operation,
  const cuDoubleComplex alpha,
  const alphasparse_matrix_t A,
  const struct alpha_matrix_descr
    descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t +
              alphasparse_diag_type_t */
  const alphasparse_layout_t
    layout, /* storage scheme for the dense matrix: C-style or Fortran-style */
  const cuDoubleComplex* x,
  const int columns,
  const int ldx,
  cuDoubleComplex* y,
  const int ldy);

/* Sparse-sparse functionality */

/*   Computes sum of sparse matrices: C = alpha * op(A) + B, result is sparse */
alphasparseStatus_t
alphasparse_s_add(const alphasparseOperation_t operation,
                  const alphasparse_matrix_t A,
                  const float alpha,
                  const alphasparse_matrix_t B,
                  alphasparse_matrix_t* C);

alphasparseStatus_t
alphasparse_d_add(const alphasparseOperation_t operation,
                  const alphasparse_matrix_t A,
                  const double alpha,
                  const alphasparse_matrix_t B,
                  alphasparse_matrix_t* C);

alphasparseStatus_t
alphasparse_c_add(const alphasparseOperation_t operation,
                  const alphasparse_matrix_t A,
                  const cuFloatComplex alpha,
                  const alphasparse_matrix_t B,
                  alphasparse_matrix_t* C);

alphasparseStatus_t
alphasparse_z_add(const alphasparseOperation_t operation,
                  const alphasparse_matrix_t A,
                  const cuDoubleComplex alpha,
                  const alphasparse_matrix_t B,
                  alphasparse_matrix_t* C);

/*   Computes product of sparse matrices: C = op(A) * B, result is sparse   */
alphasparseStatus_t
alphasparse_spmm(const alphasparseOperation_t operation,
                 const alphasparse_matrix_t A,
                 const alphasparse_matrix_t B,
                 alphasparse_matrix_t* C);

/*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse
 */
alphasparseStatus_t
alphasparse_sp2m(const alphasparseOperation_t transA,
                 const struct alpha_matrix_descr descrA,
                 const alphasparse_matrix_t A,
                 const alphasparseOperation_t transB,
                 const struct alpha_matrix_descr descrB,
                 const alphasparse_matrix_t B,
                 const alphasparse_request_t request,
                 alphasparse_matrix_t* C);

/*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H
 * for complex}, result is sparse   */
alphasparseStatus_t
alphasparse_syrk(const alphasparseOperation_t operation,
                 const alphasparse_matrix_t A,
                 alphasparse_matrix_t* C);

/*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or
 * H for complex}, result is sparse   */
alphasparseStatus_t
alphasparse_sypr(const alphasparseOperation_t transA,
                 const alphasparse_matrix_t A,
                 const alphasparse_matrix_t B,
                 const struct alpha_matrix_descr descrB,
                 alphasparse_matrix_t* C,
                 const alphasparse_request_t request);

/*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or
 * H for complex}, result is dense */
alphasparseStatus_t
alphasparse_s_syprd(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const float* B,
                    const alphasparse_layout_t layoutB,
                    const int ldb,
                    const float alpha,
                    const float beta,
                    float* C,
                    const alphasparse_layout_t layoutC,
                    const int ldc);

alphasparseStatus_t
alphasparse_d_syprd(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const double* B,
                    const alphasparse_layout_t layoutB,
                    const int ldb,
                    const double alpha,
                    const double beta,
                    double* C,
                    const alphasparse_layout_t layoutC,
                    const int ldc);

alphasparseStatus_t
alphasparse_c_syprd(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const cuFloatComplex* B,
                    const alphasparse_layout_t layoutB,
                    const int ldb,
                    const cuFloatComplex alpha,
                    const cuFloatComplex beta,
                    cuFloatComplex* C,
                    const alphasparse_layout_t layoutC,
                    const int ldc);

alphasparseStatus_t
alphasparse_z_syprd(const alphasparseOperation_t op,
                    const alphasparse_matrix_t A,
                    const cuDoubleComplex* B,
                    const alphasparse_layout_t layoutB,
                    const int ldb,
                    const cuDoubleComplex alpha,
                    const cuDoubleComplex beta,
                    cuDoubleComplex* C,
                    const alphasparse_layout_t layoutC,
                    const int ldc);

/*   Computes product of sparse matrices: C = op(A) * B, result is dense   */
alphasparseStatus_t
alphasparse_s_spmmd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const alphasparse_matrix_t B,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  float* C,
  const int ldc);

alphasparseStatus_t
alphasparse_d_spmmd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const alphasparse_matrix_t B,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  double* C,
  const int ldc);

alphasparseStatus_t
alphasparse_c_spmmd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const alphasparse_matrix_t B,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  cuFloatComplex* C,
  const int ldc);

alphasparseStatus_t
alphasparse_z_spmmd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const alphasparse_matrix_t B,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  cuDoubleComplex* C,
  const int ldc);

/*   Computes product of sparse matrices: C = opA(A) * opB(B), result is dense*/
alphasparseStatus_t
alphasparse_s_sp2md(const alphasparseOperation_t transA,
                    const struct alpha_matrix_descr descrA,
                    const alphasparse_matrix_t A,
                    const alphasparseOperation_t transB,
                    const struct alpha_matrix_descr descrB,
                    const alphasparse_matrix_t B,
                    const float alpha,
                    const float beta,
                    float* C,
                    const alphasparse_layout_t layout,
                    const int ldc);

alphasparseStatus_t
alphasparse_d_sp2md(const alphasparseOperation_t transA,
                    const struct alpha_matrix_descr descrA,
                    const alphasparse_matrix_t A,
                    const alphasparseOperation_t transB,
                    const struct alpha_matrix_descr descrB,
                    const alphasparse_matrix_t B,
                    const double alpha,
                    const double beta,
                    double* C,
                    const alphasparse_layout_t layout,
                    const int ldc);

alphasparseStatus_t
alphasparse_c_sp2md(const alphasparseOperation_t transA,
                    const struct alpha_matrix_descr descrA,
                    const alphasparse_matrix_t A,
                    const alphasparseOperation_t transB,
                    const struct alpha_matrix_descr descrB,
                    const alphasparse_matrix_t B,
                    const cuFloatComplex alpha,
                    const cuFloatComplex beta,
                    cuFloatComplex* C,
                    const alphasparse_layout_t layout,
                    const int ldc);

alphasparseStatus_t
alphasparse_z_sp2md(const alphasparseOperation_t transA,
                    const struct alpha_matrix_descr descrA,
                    const alphasparse_matrix_t A,
                    const alphasparseOperation_t transB,
                    const struct alpha_matrix_descr descrB,
                    const alphasparse_matrix_t B,
                    const cuDoubleComplex alpha,
                    const cuDoubleComplex beta,
                    cuDoubleComplex* C,
                    const alphasparse_layout_t layout,
                    const int ldc);

/*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H
 * for complex}, result is dense */
alphasparseStatus_t
alphasparse_s_syrkd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const float alpha,
  const float beta,
  float* C,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  const int ldc);

alphasparseStatus_t
alphasparse_d_syrkd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const double alpha,
  const double beta,
  double* C,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  const int ldc);

alphasparseStatus_t
alphasparse_c_syrkd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const cuFloatComplex alpha,
  const cuFloatComplex beta,
  cuFloatComplex* C,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  const int ldc);

alphasparseStatus_t
alphasparse_z_syrkd(
  const alphasparseOperation_t operation,
  const alphasparse_matrix_t A,
  const cuDoubleComplex alpha,
  const cuDoubleComplex beta,
  cuDoubleComplex* C,
  const alphasparse_layout_t layout, /* storage scheme for the output dense
                                        matrix: C-style or Fortran-style */
  const int ldc);

alphasparseStatus_t
alphasparse_s_axpy(const int nz,
                   const float a,
                   const float* x,
                   const int* indx,
                   float* y);

alphasparseStatus_t
alphasparse_d_axpy(const int nz,
                   const double a,
                   const double* x,
                   const int* indx,
                   double* y);

alphasparseStatus_t
alphasparse_c_axpy(const int nz,
                   const cuFloatComplex a,
                   const cuFloatComplex* x,
                   const int* indx,
                   cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_axpy(const int nz,
                   const cuDoubleComplex a,
                   const cuDoubleComplex* x,
                   const int* indx,
                   cuDoubleComplex* y);

alphasparseStatus_t
alphasparse_s_gthr(const int nz, const float* y, float* x, const int* indx);

alphasparseStatus_t
alphasparse_d_gthr(const int nz, const double* y, double* x, const int* indx);

alphasparseStatus_t
alphasparse_c_gthr(const int nz,
                   const cuFloatComplex* y,
                   cuFloatComplex* x,
                   const int* indx);

alphasparseStatus_t
alphasparse_z_gthr(const int nz,
                   const cuDoubleComplex* y,
                   cuDoubleComplex* x,
                   const int* indx);

alphasparseStatus_t
alphasparse_s_gthrz(const int nz, float* y, float* x, const int* indx);

alphasparseStatus_t
alphasparse_d_gthrz(const int nz, double* y, double* x, const int* indx);

alphasparseStatus_t
alphasparse_c_gthrz(const int nz,
                    cuFloatComplex* y,
                    cuFloatComplex* x,
                    const int* indx);

alphasparseStatus_t
alphasparse_z_gthrz(const int nz,
                    cuDoubleComplex* y,
                    cuDoubleComplex* x,
                    const int* indx);

alphasparseStatus_t
alphasparse_s_rot(const int nz,
                  float* x,
                  const int* indx,
                  float* y,
                  const float c,
                  const float s);

alphasparseStatus_t
alphasparse_d_rot(const int nz,
                  double* x,
                  const int* indx,
                  double* y,
                  const double c,
                  const double s);

alphasparseStatus_t
alphasparse_s_sctr(const int nz, const float* x, const int* indx, float* y);

alphasparseStatus_t
alphasparse_d_sctr(const int nz, const double* x, const int* indx, double* y);

alphasparseStatus_t
alphasparse_c_sctr(const int nz,
                   const cuFloatComplex* x,
                   const int* indx,
                   cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_sctr(const int nz,
                   const cuDoubleComplex* x,
                   const int* indx,
                   cuDoubleComplex* y);

float
alphasparse_s_doti(const int nz,
                   const float* x,
                   const int* indx,
                   const float* y);

double
alphasparse_d_doti(const int nz,
                   const double* x,
                   const int* indx,
                   const double* y);

void
alphasparse_c_dotci_sub(const int nz,
                        const cuFloatComplex* x,
                        const int* indx,
                        const cuFloatComplex* y,
                        cuFloatComplex* dutci);

void
alphasparse_z_dotci_sub(const int nz,
                        const cuDoubleComplex* x,
                        const int* indx,
                        const cuDoubleComplex* y,
                        cuDoubleComplex* dutci);

void
alphasparse_c_dotui_sub(const int nz,
                        const cuFloatComplex* x,
                        const int* indx,
                        const cuFloatComplex* y,
                        cuFloatComplex* dutui);

void
alphasparse_z_dotui_sub(const int nz,
                        const cuDoubleComplex* x,
                        const int* indx,
                        const cuDoubleComplex* y,
                        cuDoubleComplex* dutui);

#ifndef __HYGON__ 
#include "handle.h"
alphasparseStatus_t
alphasparse_s_axpyi(alphasparseHandle_t handle,
                    int nnz,
                    const float* alpha,
                    const float* x_val,
                    const int* x_ind,
                    float* y,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_d_axpyi(alphasparseHandle_t handle,
                    int nnz,
                    const double* alpha,
                    const double* x_val,
                    const int* x_ind,
                    double* y,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_c_axpyi(alphasparseHandle_t handle,
                    int nnz,
                    const cuFloatComplex* alpha,
                    const cuFloatComplex* x_val,
                    const int* x_ind,
                    cuFloatComplex* y,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_z_axpyi(alphasparseHandle_t handle,
                    int nnz,
                    const cuDoubleComplex* alpha,
                    const cuDoubleComplex* x_val,
                    const int* x_ind,
                    cuDoubleComplex* y,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_s_doti(alphasparseHandle_t handle,
                   int nnz,
                   const float* x_val,
                   const int* x_ind,
                   const float* y,
                   float* result,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_d_doti(alphasparseHandle_t handle,
                   int nnz,
                   const double* x_val,
                   const int* x_ind,
                   const double* y,
                   double* result,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_c_doti(alphasparseHandle_t handle,
                   int nnz,
                   const cuFloatComplex* x_val,
                   const int* x_ind,
                   const cuFloatComplex* y,
                   cuFloatComplex* result,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_z_doti(alphasparseHandle_t handle,
                   int nnz,
                   const cuDoubleComplex* x_val,
                   const int* x_ind,
                   const cuDoubleComplex* y,
                   cuDoubleComplex* result,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_c_dotci(alphasparseHandle_t handle,
                    int nnz,
                    const cuFloatComplex* x_val,
                    const int* x_ind,
                    const cuFloatComplex* y,
                    cuFloatComplex* result,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_z_dotci(alphasparseHandle_t handle,
                    int nnz,
                    const cuDoubleComplex* x_val,
                    const int* x_ind,
                    const cuDoubleComplex* y,
                    cuDoubleComplex* result,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_s_gthr(alphasparseHandle_t handle,
                   int nnz,
                   const float* y,
                   float* x_val,
                   const int* x_ind,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_d_gthr(alphasparseHandle_t handle,
                   int nnz,
                   const double* y,
                   double* x_val,
                   const int* x_ind,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_c_gthr(alphasparseHandle_t handle,
                   int nnz,
                   const cuFloatComplex* y,
                   cuFloatComplex* x_val,
                   const int* x_ind,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_z_gthr(alphasparseHandle_t handle,
                   int nnz,
                   const cuDoubleComplex* y,
                   cuDoubleComplex* x_val,
                   const int* x_ind,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_s_gthrz(alphasparseHandle_t handle,
                    int nnz,
                    float* y,
                    float* x_val,
                    const int* x_ind,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_d_gthrz(alphasparseHandle_t handle,
                    int nnz,
                    double* y,
                    double* x_val,
                    const int* x_ind,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_c_gthrz(alphasparseHandle_t handle,
                    int nnz,
                    cuFloatComplex* y,
                    cuFloatComplex* x_val,
                    const int* x_ind,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_z_gthrz(alphasparseHandle_t handle,
                    int nnz,
                    cuDoubleComplex* y,
                    cuDoubleComplex* x_val,
                    const int* x_ind,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_s_roti(alphasparseHandle_t handle,
                   int nnz,
                   float* x_val,
                   const int* x_ind,
                   float* y,
                   const float* c,
                   const float* s,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_d_roti(alphasparseHandle_t handle,
                   int nnz,
                   double* x_val,
                   const int* x_ind,
                   double* y,
                   const double* c,
                   const double* s,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_s_sctr(alphasparseHandle_t handle,
                   int nnz,
                   const float* x_val,
                   const int* x_ind,
                   float* y,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_d_sctr(alphasparseHandle_t handle,
                   int nnz,
                   const double* x_val,
                   const int* x_ind,
                   double* y,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_c_sctr(alphasparseHandle_t handle,
                   int nnz,
                   const cuFloatComplex* x_val,
                   const int* x_ind,
                   cuFloatComplex* y,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_z_sctr(alphasparseHandle_t handle,
                   int nnz,
                   const cuDoubleComplex* x_val,
                   const int* x_ind,
                   cuDoubleComplex* y,
                   alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_s_bsrmv(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    alphasparseOperation_t trans,
                    int mb,
                    int nb,
                    int nnzb,
                    const float* alpha,
                    const alpha_matrix_descr_t descr,
                    const float* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int bsr_dim,
                    const float* x,
                    const float* beta,
                    float* y);

alphasparseStatus_t
alphasparse_d_bsrmv(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    alphasparseOperation_t trans,
                    int mb,
                    int nb,
                    int nnzb,
                    const double* alpha,
                    const alpha_matrix_descr_t descr,
                    const double* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int bsr_dim,
                    const double* x,
                    const double* beta,
                    double* y);

alphasparseStatus_t
alphasparse_c_bsrmv(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    alphasparseOperation_t trans,
                    int mb,
                    int nb,
                    int nnzb,
                    const cuFloatComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int bsr_dim,
                    const cuFloatComplex* x,
                    const cuFloatComplex* beta,
                    cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_bsrmv(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    alphasparseOperation_t trans,
                    int mb,
                    int nb,
                    int nnzb,
                    const cuDoubleComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int bsr_dim,
                    const cuDoubleComplex* x,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* y);

alphasparseStatus_t
alphasparseSbsrmv(alphasparseHandle_t handle,
                  alphasparseDirection_t dir,
                  alphasparseOperation_t trans,
                  int mb,
                  int nb,
                  int nnzb,
                  const float* alpha,
                  const alphasparseMatDescr_t descr,
                  const float* bsrVal,
                  const int* bsrRowPtr,
                  const int* bsrColInd,
                  int blockDim,
                  const float* x,
                  const float* beta,
                  float* y);

alphasparseStatus_t
alphasparseDbsrmv(alphasparseHandle_t handle,
                  alphasparseDirection_t dir,
                  alphasparseOperation_t trans,
                  int mb,
                  int nb,
                  int nnzb,
                  const double* alpha,
                  const alphasparseMatDescr_t descr,
                  const double* bsrVal,
                  const int* bsrRowPtr,
                  const int* bsrColInd,
                  int blockDim,
                  const double* x,
                  const double* beta,
                  double* y);

alphasparseStatus_t
alphasparseCbsrmv(alphasparseHandle_t handle,
                  alphasparseDirection_t dir,
                  alphasparseOperation_t trans,
                  int mb,
                  int nb,
                  int nnzb,
                  const void* alpha,
                  const alphasparseMatDescr_t descr,
                  const void* bsrVal,
                  const int* bsrRowPtr,
                  const int* bsrColInd,
                  int blockDim,
                  const void* x,
                  const void* beta,
                  void* y);

alphasparseStatus_t
alphasparseZbsrmv(alphasparseHandle_t handle,
                  alphasparseDirection_t dir,
                  alphasparseOperation_t trans,
                  int mb,
                  int nb,
                  int nnzb,
                  const void* alpha,
                  const alphasparseMatDescr_t descr,
                  const void* bsrVal,
                  const int* bsrRowPtr,
                  const int* bsrColInd,
                  int blockDim,
                  const void* x,
                  const void* beta,
                  void* y);

alphasparseStatus_t
alphasparseSbsrxmv(alphasparseHandle_t handle,
                   alphasparseDirection_t dir,
                   alphasparseOperation_t trans,
                   int sizeOfMask,
                   int mb,
                   int nb,
                   int nnzb,
                   const float* alpha,
                   const alphasparseMatDescr_t descr,
                   const float* bsrVal,
                   const int* bsrMaskPtr,
                   const int* bsrRowPtr,
                   const int* bsrEndPtr,
                   const int* bsrColInd,
                   int blockDim,
                   const float* x,
                   const float* beta,
                   float* y);

alphasparseStatus_t
alphasparseDbsrxmv(alphasparseHandle_t handle,
                   alphasparseDirection_t dir,
                   alphasparseOperation_t trans,
                   int sizeOfMask,
                   int mb,
                   int nb,
                   int nnzb,
                   const double* alpha,
                   const alphasparseMatDescr_t descr,
                   const double* bsrVal,
                   const int* bsrMaskPtr,
                   const int* bsrRowPtr,
                   const int* bsrEndPtr,
                   const int* bsrColInd,
                   int blockDim,
                   const double* x,
                   const double* beta,
                   double* y);

alphasparseStatus_t
alphasparseCbsrxmv(alphasparseHandle_t handle,
                   alphasparseDirection_t dir,
                   alphasparseOperation_t trans,
                   int sizeOfMask,
                   int mb,
                   int nb,
                   int nnzb,
                   const void* alpha,
                   const alphasparseMatDescr_t descr,
                   const void* bsrVal,
                   const int* bsrMaskPtr,
                   const int* bsrRowPtr,
                   const int* bsrEndPtr,
                   const int* bsrColInd,
                   int blockDim,
                   const void* x,
                   const void* beta,
                   void* y);

alphasparseStatus_t
alphasparseZbsrxmv(alphasparseHandle_t handle,
                   alphasparseDirection_t dir,
                   alphasparseOperation_t trans,
                   int sizeOfMask,
                   int mb,
                   int nb,
                   int nnzb,
                   const void* alpha,
                   const alphasparseMatDescr_t descr,
                   const void* bsrVal,
                   const int* bsrMaskPtr,
                   const int* bsrRowPtr,
                   const int* bsrEndPtr,
                   const int* bsrColInd,
                   int blockDim,
                   const void* x,
                   const void* beta,
                   void* y);

alphasparseStatus_t
alphasparseSbsrsv2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const float* alpha,
                         const alphasparseMatDescr_t descrA,
                         const float* bsrValA,
                         const int* bsrRowPtrA,
                         const int* bsrColIndA,
                         int blockDim,
                         alpha_bsrsv2Info_t info,
                         const float* x,
                         float* y,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseDbsrsv2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const double* alpha,
                         const alphasparseMatDescr_t descrA,
                         const double* bsrValA,
                         const int* bsrRowPtrA,
                         const int* bsrColIndA,
                         int blockDim,
                         alpha_bsrsv2Info_t info,
                         const double* x,
                         double* y,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseCbsrsv2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const void* alpha,
                         const alphasparseMatDescr_t descrA,
                         const void* bsrValA,
                         const int* bsrRowPtrA,
                         const int* bsrColIndA,
                         int blockDim,
                         alpha_bsrsv2Info_t info,
                         const void* x,
                         void* y,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseZbsrsv2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         int mb,
                         int nnzb,
                         const void* alpha,
                         const alphasparseMatDescr_t descrA,
                         const void* bsrValA,
                         const int* bsrRowPtrA,
                         const int* bsrColIndA,
                         int blockDim,
                         alpha_bsrsv2Info_t info,
                         const void* x,
                         void* y,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseSbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              float* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              double* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZbsrsv2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrsv2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSbsrsv2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            float* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrsv2Info_t info,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDbsrsv2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            double* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrsv2Info_t info,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCbsrsv2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrsv2Info_t info,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZbsrsv2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            int mb,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrValA,
                            const int* bsrRowPtrA,
                            const int* bsrColIndA,
                            int blockDim,
                            alpha_bsrsv2Info_t info,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSgemvi(alphasparseHandle_t handle,
                  alphasparseOperation_t transA,
                  int m,
                  int n,
                  const float* alpha,
                  const float* A,
                  int lda,
                  int nnz,
                  const float* x,
                  const int* xInd,
                  const float* beta,
                  float* y,
                  alphasparseIndexBase_t idxBase,
                  void* pBuffer);

alphasparseStatus_t
alphasparseDgemvi(alphasparseHandle_t handle,
                  alphasparseOperation_t transA,
                  int m,
                  int n,
                  const double* alpha,
                  const double* A,
                  int lda,
                  int nnz,
                  const double* x,
                  const int* xInd,
                  const double* beta,
                  double* y,
                  alphasparseIndexBase_t idxBase,
                  void* pBuffer);

alphasparseStatus_t
alphasparseCgemvi(alphasparseHandle_t handle,
                  alphasparseOperation_t transA,
                  int m,
                  int n,
                  const void* alpha,
                  const void* A,
                  int lda,
                  int nnz,
                  const void* x,
                  const int* xInd,
                  const void* beta,
                  void* y,
                  alphasparseIndexBase_t idxBase,
                  void* pBuffer);

alphasparseStatus_t
alphasparseZgemvi(alphasparseHandle_t handle,
                  alphasparseOperation_t transA,
                  int m,
                  int n,
                  const void* alpha,
                  const void* A,
                  int lda,
                  int nnz,
                  const void* x,
                  const int* xInd,
                  const void* beta,
                  void* y,
                  alphasparseIndexBase_t idxBase,
                  void* pBuffer);

alphasparseStatus_t
alphasparseSbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const float* alpha,
                  const alphasparseMatDescr_t descrA,
                  const float* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const float* B,
                  int ldb,
                  const float* beta,
                  float* C,
                  int ldc);

alphasparseStatus_t
alphasparseDbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const double* alpha,
                  const alphasparseMatDescr_t descrA,
                  const double* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const double* B,
                  int ldb,
                  const double* beta,
                  double* C,
                  int ldc);

alphasparseStatus_t
alphasparseCbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const void* alpha,
                  const alphasparseMatDescr_t descrA,
                  const void* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const void* B,
                  int ldb,
                  const void* beta,
                  void* C,
                  int ldc);

alphasparseStatus_t
alphasparseZbsrmm(alphasparseHandle_t handle,
                  alphasparseDirection_t dirA,
                  alphasparseOperation_t transA,
                  alphasparseOperation_t transB,
                  int mb,
                  int n,
                  int kb,
                  int nnzb,
                  const void* alpha,
                  const alphasparseMatDescr_t descrA,
                  const void* bsrValA,
                  const int* bsrRowPtrA,
                  const int* bsrColIndA,
                  int blockDim,
                  const void* B,
                  int ldb,
                  const void* beta,
                  void* C,
                  int ldc);

alphasparseStatus_t
alphasparse_bsrsv_zero_pivot(alphasparseHandle_t handle,
                             alphasparse_mat_info_t info,
                             int* position);

alphasparseStatus_t
alphasparse_s_bsrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                alphasparseOperation_t trans,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const float* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int bsr_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_d_bsrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                alphasparseOperation_t trans,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const double* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int bsr_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_c_bsrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                alphasparseOperation_t trans,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const cuFloatComplex* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int bsr_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_z_bsrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                alphasparseOperation_t trans,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const cuDoubleComplex* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int bsr_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_s_bsrsv_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             alphasparseOperation_t trans,
                             int mb,
                             int nnzb,
                             const float* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int bsr_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_d_bsrsv_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             alphasparseOperation_t trans,
                             int mb,
                             int nnzb,
                             const double* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int bsr_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_c_bsrsv_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             alphasparseOperation_t trans,
                             int mb,
                             int nnzb,
                             const alpha_matrix_descr_t descr,
                             const cuFloatComplex* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int bsr_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_z_bsrsv_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             alphasparseOperation_t trans,
                             int mb,
                             int nnzb,
                             const alpha_matrix_descr_t descr,
                             const cuDoubleComplex* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int bsr_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_bsrsv_clear(alphasparseHandle_t handle,
                        alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_s_bsrsv_solve(alphasparseHandle_t handle,
                          alphasparse_layout_t dir,
                          alphasparseOperation_t trans,
                          int mb,
                          int nnzb,
                          const float* alpha,
                          const alpha_matrix_descr_t descr,
                          const float* bsr_val,
                          const int* bsr_row_ptr,
                          const int* bsr_col_ind,
                          int bsr_dim,
                          alphasparse_mat_info_t info,
                          const float* x,
                          float* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_d_bsrsv_solve(alphasparseHandle_t handle,
                          alphasparse_layout_t dir,
                          alphasparseOperation_t trans,
                          int mb,
                          int nnzb,
                          const double* alpha,
                          const alpha_matrix_descr_t descr,
                          const double* bsr_val,
                          const int* bsr_row_ptr,
                          const int* bsr_col_ind,
                          int bsr_dim,
                          alphasparse_mat_info_t info,
                          const double* x,
                          double* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_c_bsrsv_solve(alphasparseHandle_t handle,
                          alphasparse_layout_t dir,
                          alphasparseOperation_t trans,
                          int mb,
                          int nnzb,
                          const cuFloatComplex* alpha,
                          const alpha_matrix_descr_t descr,
                          const cuFloatComplex* bsr_val,
                          const int* bsr_row_ptr,
                          const int* bsr_col_ind,
                          int bsr_dim,
                          alphasparse_mat_info_t info,
                          const cuFloatComplex* x,
                          cuFloatComplex* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_z_bsrsv_solve(alphasparseHandle_t handle,
                          alphasparse_layout_t dir,
                          alphasparseOperation_t trans,
                          int mb,
                          int nnzb,
                          const cuDoubleComplex* alpha,
                          const alpha_matrix_descr_t descr,
                          const cuDoubleComplex* bsr_val,
                          const int* bsr_row_ptr,
                          const int* bsr_col_ind,
                          int bsr_dim,
                          alphasparse_mat_info_t info,
                          const cuDoubleComplex* x,
                          cuDoubleComplex* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_s_coomv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const float* alpha,
                    const alpha_matrix_descr_t descr,
                    const float* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const float* x,
                    const float* beta,
                    float* y);

alphasparseStatus_t
alphasparse_d_coomv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const double* alpha,
                    const alpha_matrix_descr_t descr,
                    const double* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const double* x,
                    const double* beta,
                    double* y);

alphasparseStatus_t
alphasparse_c_coomv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const cuFloatComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const cuFloatComplex* x,
                    const cuFloatComplex* beta,
                    cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_coomv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const cuDoubleComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const cuDoubleComplex* x,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* y);

alphasparseStatus_t
alphasparse_s_csrmv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int n,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const float* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_d_csrmv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int n,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const double* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_c_csrmv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int n,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const cuFloatComplex* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_z_csrmv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int n,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const cuDoubleComplex* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_csrmv_clear(alphasparseHandle_t handle,
                        alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_s_csrmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const float* alpha,
                    const alpha_matrix_descr_t descr,
                    const float* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    const float* x,
                    const float* beta,
                    float* y);

alphasparseStatus_t
alphasparse_d_csrmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const double* alpha,
                    const alpha_matrix_descr_t descr,
                    const double* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    const double* x,
                    const double* beta,
                    double* y);

alphasparseStatus_t
alphasparse_c_csrmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const cuFloatComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    const cuFloatComplex* x,
                    const cuFloatComplex* beta,
                    cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_csrmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    int nnz,
                    const cuDoubleComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    const cuDoubleComplex* x,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* y);

alphasparseStatus_t
alphasparse_csrsv_zero_pivot(alphasparseHandle_t handle,
                             const alpha_matrix_descr_t descr,
                             alphasparse_mat_info_t info,
                             int* position);

alphasparseStatus_t
alphasparse_s_csrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparseOperation_t trans,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const float* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_d_csrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparseOperation_t trans,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const double* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_c_csrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparseOperation_t trans,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const cuFloatComplex* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_z_csrsv_buffer_size(alphasparseHandle_t handle,
                                alphasparseOperation_t trans,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const cuDoubleComplex* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_s_csrsv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const float* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_d_csrsv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const double* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_c_csrsv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const cuFloatComplex* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_z_csrsv_analysis(alphasparseHandle_t handle,
                             alphasparseOperation_t trans,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const cuDoubleComplex* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_csrsv_clear(alphasparseHandle_t handle,
                        const alpha_matrix_descr_t descr,
                        alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_s_csrsv_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans,
                          int m,
                          int nnz,
                          const float* alpha,
                          const alpha_matrix_descr_t descr,
                          const float* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          alphasparse_mat_info_t info,
                          const float* x,
                          float* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_d_csrsv_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans,
                          int m,
                          int nnz,
                          const double* alpha,
                          const alpha_matrix_descr_t descr,
                          const double* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          alphasparse_mat_info_t info,
                          const double* x,
                          double* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_c_csrsv_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans,
                          int m,
                          int nnz,
                          const cuFloatComplex* alpha,
                          const alpha_matrix_descr_t descr,
                          const cuFloatComplex* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          alphasparse_mat_info_t info,
                          const cuFloatComplex* x,
                          cuFloatComplex* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_z_csrsv_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans,
                          int m,
                          int nnz,
                          const cuDoubleComplex* alpha,
                          const alpha_matrix_descr_t descr,
                          const cuDoubleComplex* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          alphasparse_mat_info_t info,
                          const cuDoubleComplex* x,
                          cuDoubleComplex* y,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_s_ellmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    const float* alpha,
                    const alpha_matrix_descr_t descr,
                    const float* ell_val,
                    const int* ell_col_ind,
                    int ell_width,
                    const float* x,
                    const float* beta,
                    float* y);

alphasparseStatus_t
alphasparse_d_ellmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    const double* alpha,
                    const alpha_matrix_descr_t descr,
                    const double* ell_val,
                    const int* ell_col_ind,
                    int ell_width,
                    const double* x,
                    const double* beta,
                    double* y);

alphasparseStatus_t
alphasparse_c_ellmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    const cuFloatComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* ell_val,
                    const int* ell_col_ind,
                    int ell_width,
                    const cuFloatComplex* x,
                    const cuFloatComplex* beta,
                    cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_ellmv(alphasparseHandle_t handle,
                    alphasparseOperation_t trans,
                    int m,
                    int n,
                    const cuDoubleComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* ell_val,
                    const int* ell_col_ind,
                    int ell_width,
                    const cuDoubleComplex* x,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* y);

alphasparseStatus_t
alphasparse_s_csrmm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const float* alpha,
                    const alpha_matrix_descr_t descr,
                    const float* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const float* B,
                    int ldb,
                    const float* beta,
                    float* C,
                    int ldc);

alphasparseStatus_t
alphasparse_d_csrmm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const double* alpha,
                    const alpha_matrix_descr_t descr,
                    const double* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const double* B,
                    int ldb,
                    const double* beta,
                    double* C,
                    int ldc);

alphasparseStatus_t
alphasparse_c_csrmm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const cuFloatComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const cuFloatComplex* B,
                    int ldb,
                    const cuFloatComplex* beta,
                    cuFloatComplex* C,
                    int ldc);

alphasparseStatus_t
alphasparse_z_csrmm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const cuDoubleComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const cuDoubleComplex* B,
                    int ldb,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* C,
                    int ldc);

/**
 * coomm
 *
 */
alphasparseStatus_t
alphasparse_s_coomm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const float* alpha,
                    const alpha_matrix_descr_t descr,
                    const float* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const float* B,
                    int ldb,
                    const float* beta,
                    float* C,
                    int ldc);

alphasparseStatus_t
alphasparse_d_coomm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const double* alpha,
                    const alpha_matrix_descr_t descr,
                    const double* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const double* B,
                    int ldb,
                    const double* beta,
                    double* C,
                    int ldc);

alphasparseStatus_t
alphasparse_c_coomm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const cuFloatComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const cuFloatComplex* B,
                    int ldb,
                    const cuFloatComplex* beta,
                    cuFloatComplex* C,
                    int ldc);

alphasparseStatus_t
alphasparse_z_coomm(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    alphasparse_layout_t layout,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const cuDoubleComplex* alpha,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* coo_val,
                    const int* coo_row_ind,
                    const int* coo_col_ind,
                    const cuDoubleComplex* B,
                    int ldb,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* C,
                    int ldc);

alphasparseStatus_t
alphasparse_csrsm_zero_pivot(alphasparseHandle_t handle,
                             alphasparse_mat_info_t info,
                             int* position);

alphasparseStatus_t
alphasparseSbsrsm2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         alphasparseOperation_t transX,
                         int mb,
                         int n,
                         int nnzb,
                         const float* alpha,
                         const alphasparseMatDescr_t descrA,
                         const float* bsrSortedVal,
                         const int* bsrSortedRowPtr,
                         const int* bsrSortedColInd,
                         int blockDim,
                         alpha_bsrsm2Info_t info,
                         const float* B,
                         int ldb,
                         float* X,
                         int ldx,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseDbsrsm2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         alphasparseOperation_t transX,
                         int mb,
                         int n,
                         int nnzb,
                         const double* alpha,
                         const alphasparseMatDescr_t descrA,
                         const double* bsrSortedVal,
                         const int* bsrSortedRowPtr,
                         const int* bsrSortedColInd,
                         int blockDim,
                         alpha_bsrsm2Info_t info,
                         const double* B,
                         int ldb,
                         double* X,
                         int ldx,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseCbsrsm2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         alphasparseOperation_t transX,
                         int mb,
                         int n,
                         int nnzb,
                         const void* alpha,
                         const alphasparseMatDescr_t descrA,
                         const void* bsrSortedVal,
                         const int* bsrSortedRowPtr,
                         const int* bsrSortedColInd,
                         int blockDim,
                         alpha_bsrsm2Info_t info,
                         const void* B,
                         int ldb,
                         void* X,
                         int ldx,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseZbsrsm2_solve(alphasparseHandle_t handle,
                         alphasparseDirection_t dirA,
                         alphasparseOperation_t transA,
                         alphasparseOperation_t transX,
                         int mb,
                         int n,
                         int nnzb,
                         const void* alpha,
                         const alphasparseMatDescr_t descrA,
                         const void* bsrSortedVal,
                         const int* bsrSortedRowPtr,
                         const int* bsrSortedColInd,
                         int blockDim,
                         alpha_bsrsm2Info_t info,
                         const void* B,
                         int ldb,
                         void* X,
                         int ldx,
                         alphasparseSolvePolicy_t policy,
                         void* pBuffer);

alphasparseStatus_t
alphasparseSbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              float* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              double* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZbsrsm2_bufferSize(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              alphasparseOperation_t transA,
                              alphasparseOperation_t transX,
                              int mb,
                              int n,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              void* bsrSortedValA,
                              const int* bsrSortedRowPtrA,
                              const int* bsrSortedColIndA,
                              int blockDim,
                              alpha_bsrsm2Info_t info,
                              int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            float* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            double* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZbsrsm2_analysis(alphasparseHandle_t handle,
                            alphasparseDirection_t dirA,
                            alphasparseOperation_t transA,
                            alphasparseOperation_t transX,
                            int mb,
                            int n,
                            int nnzb,
                            const alphasparseMatDescr_t descrA,
                            void* bsrSortedValA,
                            const int* bsrSortedRowPtrA,
                            const int* bsrSortedColIndA,
                            int blockDim,
                            alpha_bsrsm2Info_t info,
                            alphasparseSolvePolicy_t policy,
                            int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseScsrgeam2(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const float* alpha,
                     const alphasparseMatDescr_t descrA,
                     int nnzA,
                     const float* csrSortedValA,
                     const int* csrSortedRowPtrA,
                     const int* csrSortedColIndA,
                     const float* beta,
                     const alphasparseMatDescr_t descrB,
                     int nnzB,
                     const float* csrSortedValB,
                     const int* csrSortedRowPtrB,
                     const int* csrSortedColIndB,
                     const alphasparseMatDescr_t descrC,
                     float* csrSortedValC,
                     int* csrSortedRowPtrC,
                     int* csrSortedColIndC,
                     void* pBuffer);

alphasparseStatus_t
alphasparseDcsrgeam2(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const double* alpha,
                     const alphasparseMatDescr_t descrA,
                     int nnzA,
                     const double* csrSortedValA,
                     const int* csrSortedRowPtrA,
                     const int* csrSortedColIndA,
                     const double* beta,
                     const alphasparseMatDescr_t descrB,
                     int nnzB,
                     const double* csrSortedValB,
                     const int* csrSortedRowPtrB,
                     const int* csrSortedColIndB,
                     const alphasparseMatDescr_t descrC,
                     double* csrSortedValC,
                     int* csrSortedRowPtrC,
                     int* csrSortedColIndC,
                     void* pBuffer);

alphasparseStatus_t
alphasparseCcsrgeam2(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const void* alpha,
                     const alphasparseMatDescr_t descrA,
                     int nnzA,
                     const void* csrSortedValA,
                     const int* csrSortedRowPtrA,
                     const int* csrSortedColIndA,
                     const void* beta,
                     const alphasparseMatDescr_t descrB,
                     int nnzB,
                     const void* csrSortedValB,
                     const int* csrSortedRowPtrB,
                     const int* csrSortedColIndB,
                     const alphasparseMatDescr_t descrC,
                     void* csrSortedValC,
                     int* csrSortedRowPtrC,
                     int* csrSortedColIndC,
                     void* pBuffer);

alphasparseStatus_t
alphasparseZcsrgeam2(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const void* alpha,
                     const alphasparseMatDescr_t descrA,
                     int nnzA,
                     const void* csrSortedValA,
                     const int* csrSortedRowPtrA,
                     const int* csrSortedColIndA,
                     const void* beta,
                     const alphasparseMatDescr_t descrB,
                     int nnzB,
                     const void* csrSortedValB,
                     const int* csrSortedRowPtrB,
                     const int* csrSortedColIndB,
                     const alphasparseMatDescr_t descrC,
                     void* csrSortedValC,
                     int* csrSortedRowPtrC,
                     int* csrSortedColIndC,
                     void* pBuffer);

alphasparseStatus_t
alphasparseXcsrgeam2Nnz(alphasparseHandle_t handle,
                        int m,
                        int n,
                        const alphasparseMatDescr_t descrA,
                        int nnzA,
                        const int* csrSortedRowPtrA,
                        const int* csrSortedColIndA,
                        const alphasparseMatDescr_t descrB,
                        int nnzB,
                        const int* csrSortedRowPtrB,
                        const int* csrSortedColIndB,
                        const alphasparseMatDescr_t descrC,
                        int* csrSortedRowPtrC,
                        int* nnzTotalDevHostPtr,
                        void* workspace);

alphasparseStatus_t
alphasparseScsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                   int m,
                                   int n,
                                   const float* alpha,
                                   const alphasparseMatDescr_t descrA,
                                   int nnzA,
                                   const float* csrSortedValA,
                                   const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA,
                                   const float* beta,
                                   const alphasparseMatDescr_t descrB,
                                   int nnzB,
                                   const float* csrSortedValB,
                                   const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB,
                                   const alphasparseMatDescr_t descrC,
                                   const float* csrSortedValC,
                                   const int* csrSortedRowPtrC,
                                   const int* csrSortedColIndC,
                                   size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDcsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                   int m,
                                   int n,
                                   const double* alpha,
                                   const alphasparseMatDescr_t descrA,
                                   int nnzA,
                                   const double* csrSortedValA,
                                   const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA,
                                   const double* beta,
                                   const alphasparseMatDescr_t descrB,
                                   int nnzB,
                                   const double* csrSortedValB,
                                   const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB,
                                   const alphasparseMatDescr_t descrC,
                                   const double* csrSortedValC,
                                   const int* csrSortedRowPtrC,
                                   const int* csrSortedColIndC,
                                   size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCcsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                   int m,
                                   int n,
                                   const void* alpha,
                                   const alphasparseMatDescr_t descrA,
                                   int nnzA,
                                   const void* csrSortedValA,
                                   const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA,
                                   const void* beta,
                                   const alphasparseMatDescr_t descrB,
                                   int nnzB,
                                   const void* csrSortedValB,
                                   const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB,
                                   const alphasparseMatDescr_t descrC,
                                   const void* csrSortedValC,
                                   const int* csrSortedRowPtrC,
                                   const int* csrSortedColIndC,
                                   size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZcsrgeam2_bufferSizeExt(alphasparseHandle_t handle,
                                   int m,
                                   int n,
                                   const void* alpha,
                                   const alphasparseMatDescr_t descrA,
                                   int nnzA,
                                   const void* csrSortedValA,
                                   const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA,
                                   const void* beta,
                                   const alphasparseMatDescr_t descrB,
                                   int nnzB,
                                   const void* csrSortedValB,
                                   const int* csrSortedRowPtrB,
                                   const int* csrSortedColIndB,
                                   const alphasparseMatDescr_t descrC,
                                   const void* csrSortedValC,
                                   const int* csrSortedRowPtrC,
                                   const int* csrSortedColIndC,
                                   size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparse_scsrsm_buffer_size(alphasparseHandle_t handle,
                               alphasparseOperation_t trans_A,
                               alphasparseOperation_t trans_B,
                               int m,
                               int nrhs,
                               int nnz,
                               const float* alpha,
                               const alpha_matrix_descr_t descr,
                               const float* csr_val,
                               const int* csr_row_ptr,
                               const int* csr_col_ind,
                               const float* B,
                               int ldb,
                               alphasparse_mat_info_t info,
                               alphasparse_solve_policy_t policy,
                               size_t* buffer_size);

alphasparseStatus_t
alphasparse_dcsrsm_buffer_size(alphasparseHandle_t handle,
                               alphasparseOperation_t trans_A,
                               alphasparseOperation_t trans_B,
                               int m,
                               int nrhs,
                               int nnz,
                               const double* alpha,
                               const alpha_matrix_descr_t descr,
                               const double* csr_val,
                               const int* csr_row_ptr,
                               const int* csr_col_ind,
                               const double* B,
                               int ldb,
                               alphasparse_mat_info_t info,
                               alphasparse_solve_policy_t policy,
                               size_t* buffer_size);

alphasparseStatus_t
alphasparse_ccsrsm_buffer_size(alphasparseHandle_t handle,
                               alphasparseOperation_t trans_A,
                               alphasparseOperation_t trans_B,
                               int m,
                               int nrhs,
                               int nnz,
                               const cuFloatComplex* alpha,
                               const alpha_matrix_descr_t descr,
                               const cuFloatComplex* csr_val,
                               const int* csr_row_ptr,
                               const int* csr_col_ind,
                               const cuFloatComplex* B,
                               int ldb,
                               alphasparse_mat_info_t info,
                               alphasparse_solve_policy_t policy,
                               size_t* buffer_size);

alphasparseStatus_t
alphasparse_zcsrsm_buffer_size(alphasparseHandle_t handle,
                               alphasparseOperation_t trans_A,
                               alphasparseOperation_t trans_B,
                               int m,
                               int nrhs,
                               int nnz,
                               const cuDoubleComplex* alpha,
                               const alpha_matrix_descr_t descr,
                               const cuDoubleComplex* csr_val,
                               const int* csr_row_ptr,
                               const int* csr_col_ind,
                               const cuDoubleComplex* B,
                               int ldb,
                               alphasparse_mat_info_t info,
                               alphasparse_solve_policy_t policy,
                               size_t* buffer_size);

alphasparseStatus_t
alphasparse_scsrsm_analysis(alphasparseHandle_t handle,
                            alphasparseOperation_t trans_A,
                            alphasparseOperation_t trans_B,
                            int m,
                            int nrhs,
                            int nnz,
                            const float* alpha,
                            const alpha_matrix_descr_t descr,
                            const float* csr_val,
                            const int* csr_row_ptr,
                            const int* csr_col_ind,
                            const float* B,
                            int ldb,
                            alphasparse_mat_info_t info,
                            alphasparse_analysis_policy_t analysis,
                            alphasparse_solve_policy_t solve,
                            void* temp_buffer);

alphasparseStatus_t
alphasparse_dcsrsm_analysis(alphasparseHandle_t handle,
                            alphasparseOperation_t trans_A,
                            alphasparseOperation_t trans_B,
                            int m,
                            int nrhs,
                            int nnz,
                            const double* alpha,
                            const alpha_matrix_descr_t descr,
                            const double* csr_val,
                            const int* csr_row_ptr,
                            const int* csr_col_ind,
                            const double* B,
                            int ldb,
                            alphasparse_mat_info_t info,
                            alphasparse_analysis_policy_t analysis,
                            alphasparse_solve_policy_t solve,
                            void* temp_buffer);

alphasparseStatus_t
alphasparse_ccsrsm_analysis(alphasparseHandle_t handle,
                            alphasparseOperation_t trans_A,
                            alphasparseOperation_t trans_B,
                            int m,
                            int nrhs,
                            int nnz,
                            const cuFloatComplex* alpha,
                            const alpha_matrix_descr_t descr,
                            const cuFloatComplex* csr_val,
                            const int* csr_row_ptr,
                            const int* csr_col_ind,
                            const cuFloatComplex* B,
                            int ldb,
                            alphasparse_mat_info_t info,
                            alphasparse_analysis_policy_t analysis,
                            alphasparse_solve_policy_t solve,
                            void* temp_buffer);

alphasparseStatus_t
alphasparse_zcsrsm_analysis(alphasparseHandle_t handle,
                            alphasparseOperation_t trans_A,
                            alphasparseOperation_t trans_B,
                            int m,
                            int nrhs,
                            int nnz,
                            const cuDoubleComplex* alpha,
                            const alpha_matrix_descr_t descr,
                            const cuDoubleComplex* csr_val,
                            const int* csr_row_ptr,
                            const int* csr_col_ind,
                            const cuDoubleComplex* B,
                            int ldb,
                            alphasparse_mat_info_t info,
                            alphasparse_analysis_policy_t analysis,
                            alphasparse_solve_policy_t solve,
                            void* temp_buffer);

alphasparseStatus_t
alphasparse_csrsm_clear(alphasparseHandle_t handle,
                        alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_s_csrsm_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans_A,
                          alphasparseOperation_t trans_B,
                          int m,
                          int nrhs,
                          int nnz,
                          const float* alpha,
                          const alpha_matrix_descr_t descr,
                          const float* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          float* B,
                          int ldb,
                          alphasparse_mat_info_t info,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_d_csrsm_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans_A,
                          alphasparseOperation_t trans_B,
                          int m,
                          int nrhs,
                          int nnz,
                          const double* alpha,
                          const alpha_matrix_descr_t descr,
                          const double* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          double* B,
                          int ldb,
                          alphasparse_mat_info_t info,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_c_csrsm_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans_A,
                          alphasparseOperation_t trans_B,
                          int m,
                          int nrhs,
                          int nnz,
                          const cuFloatComplex* alpha,
                          const alpha_matrix_descr_t descr,
                          const cuFloatComplex* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          cuFloatComplex* B,
                          int ldb,
                          alphasparse_mat_info_t info,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_z_csrsm_solve(alphasparseHandle_t handle,
                          alphasparseOperation_t trans_A,
                          alphasparseOperation_t trans_B,
                          int m,
                          int nrhs,
                          int nnz,
                          const cuDoubleComplex* alpha,
                          const alpha_matrix_descr_t descr,
                          const cuDoubleComplex* csr_val,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          cuDoubleComplex* B,
                          int ldb,
                          alphasparse_mat_info_t info,
                          alphasparse_solve_policy_t policy,
                          void* temp_buffer);

alphasparseStatus_t
alphasparse_s_gemmi(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const float* alpha,
                    const float* A,
                    int lda,
                    const alpha_matrix_descr_t descr,
                    const float* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const float* beta,
                    float* C,
                    int ldc);

alphasparseStatus_t
alphasparse_d_gemmi(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const double* alpha,
                    const double* A,
                    int lda,
                    const alpha_matrix_descr_t descr,
                    const double* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const double* beta,
                    double* C,
                    int ldc);

alphasparseStatus_t
alphasparse_c_gemmi(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const cuFloatComplex* alpha,
                    const cuFloatComplex* A,
                    int lda,
                    const alpha_matrix_descr_t descr,
                    const cuFloatComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const cuFloatComplex* beta,
                    cuFloatComplex* C,
                    int ldc);

alphasparseStatus_t
alphasparse_z_gemmi(alphasparseHandle_t handle,
                    alphasparseOperation_t trans_A,
                    alphasparseOperation_t trans_B,
                    int m,
                    int n,
                    int k,
                    int nnz,
                    const cuDoubleComplex* alpha,
                    const cuDoubleComplex* A,
                    int lda,
                    const alpha_matrix_descr_t descr,
                    const cuDoubleComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    const cuDoubleComplex* beta,
                    cuDoubleComplex* C,
                    int ldc);

alphasparseStatus_t
alphasparse_csrgeam_nnz(alphasparseHandle_t handle,
                        int m,
                        int n,
                        const alpha_matrix_descr_t descr_A,
                        int nnz_A,
                        const int* csr_row_ptr_A,
                        const int* csr_col_ind_A,
                        const alpha_matrix_descr_t descr_B,
                        int nnz_B,
                        const int* csr_row_ptr_B,
                        const int* csr_col_ind_B,
                        const alpha_matrix_descr_t descr_C,
                        int* csr_row_ptr_C,
                        int* nnz_C);

alphasparseStatus_t
alphasparse_s_csrgeam(alphasparseHandle_t handle,
                      int m,
                      int n,
                      const float* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const float* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const float* beta,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const float* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const alpha_matrix_descr_t descr_C,
                      float* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C);

alphasparseStatus_t
alphasparse_d_csrgeam(alphasparseHandle_t handle,
                      int m,
                      int n,
                      const double* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const double* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const double* beta,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const double* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const alpha_matrix_descr_t descr_C,
                      double* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C);

alphasparseStatus_t
alphasparse_c_csrgeam(alphasparseHandle_t handle,
                      int m,
                      int n,
                      const cuFloatComplex* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const cuFloatComplex* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const cuFloatComplex* beta,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const cuFloatComplex* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const alpha_matrix_descr_t descr_C,
                      cuFloatComplex* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C);

alphasparseStatus_t
alphasparse_z_csrgeam(alphasparseHandle_t handle,
                      int m,
                      int n,
                      const cuDoubleComplex* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const cuDoubleComplex* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const cuDoubleComplex* beta,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const cuDoubleComplex* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const alpha_matrix_descr_t descr_C,
                      cuDoubleComplex* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C);

alphasparseStatus_t
alphasparse_scsrgemm_buffer_size(alphasparseHandle_t handle,
                                 alphasparseOperation_t trans_A,
                                 alphasparseOperation_t trans_B,
                                 int m,
                                 int n,
                                 int k,
                                 const float* alpha,
                                 const alpha_matrix_descr_t descr_A,
                                 int nnz_A,
                                 const int* csr_row_ptr_A,
                                 const int* csr_col_ind_A,
                                 const alpha_matrix_descr_t descr_B,
                                 int nnz_B,
                                 const int* csr_row_ptr_B,
                                 const int* csr_col_ind_B,
                                 const float* beta,
                                 const alpha_matrix_descr_t descr_D,
                                 int nnz_D,
                                 const int* csr_row_ptr_D,
                                 const int* csr_col_ind_D,
                                 alphasparse_mat_info_t info_C,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_dcsrgemm_buffer_size(alphasparseHandle_t handle,
                                 alphasparseOperation_t trans_A,
                                 alphasparseOperation_t trans_B,
                                 int m,
                                 int n,
                                 int k,
                                 const double* alpha,
                                 const alpha_matrix_descr_t descr_A,
                                 int nnz_A,
                                 const int* csr_row_ptr_A,
                                 const int* csr_col_ind_A,
                                 const alpha_matrix_descr_t descr_B,
                                 int nnz_B,
                                 const int* csr_row_ptr_B,
                                 const int* csr_col_ind_B,
                                 const double* beta,
                                 const alpha_matrix_descr_t descr_D,
                                 int nnz_D,
                                 const int* csr_row_ptr_D,
                                 const int* csr_col_ind_D,
                                 alphasparse_mat_info_t info_C,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_ccsrgemm_buffer_size(alphasparseHandle_t handle,
                                 alphasparseOperation_t trans_A,
                                 alphasparseOperation_t trans_B,
                                 int m,
                                 int n,
                                 int k,
                                 const cuFloatComplex* alpha,
                                 const alpha_matrix_descr_t descr_A,
                                 int nnz_A,
                                 const int* csr_row_ptr_A,
                                 const int* csr_col_ind_A,
                                 const alpha_matrix_descr_t descr_B,
                                 int nnz_B,
                                 const int* csr_row_ptr_B,
                                 const int* csr_col_ind_B,
                                 const cuFloatComplex* beta,
                                 const alpha_matrix_descr_t descr_D,
                                 int nnz_D,
                                 const int* csr_row_ptr_D,
                                 const int* csr_col_ind_D,
                                 alphasparse_mat_info_t info_C,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_zcsrgemm_buffer_size(alphasparseHandle_t handle,
                                 alphasparseOperation_t trans_A,
                                 alphasparseOperation_t trans_B,
                                 int m,
                                 int n,
                                 int k,
                                 const cuDoubleComplex* alpha,
                                 const alpha_matrix_descr_t descr_A,
                                 int nnz_A,
                                 const int* csr_row_ptr_A,
                                 const int* csr_col_ind_A,
                                 const alpha_matrix_descr_t descr_B,
                                 int nnz_B,
                                 const int* csr_row_ptr_B,
                                 const int* csr_col_ind_B,
                                 const cuDoubleComplex* beta,
                                 const alpha_matrix_descr_t descr_D,
                                 int nnz_D,
                                 const int* csr_row_ptr_D,
                                 const int* csr_col_ind_D,
                                 alphasparse_mat_info_t info_C,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_csrgemm_nnz(alphasparseHandle_t handle,
                        alphasparseOperation_t trans_A,
                        alphasparseOperation_t trans_B,
                        int m,
                        int n,
                        int k,
                        const alpha_matrix_descr_t descr_A,
                        int nnz_A,
                        const int* csr_row_ptr_A,
                        const int* csr_col_ind_A,
                        const alpha_matrix_descr_t descr_B,
                        int nnz_B,
                        const int* csr_row_ptr_B,
                        const int* csr_col_ind_B,
                        const alpha_matrix_descr_t descr_D,
                        int nnz_D,
                        const int* csr_row_ptr_D,
                        const int* csr_col_ind_D,
                        const alpha_matrix_descr_t descr_C,
                        int* csr_row_ptr_C,
                        int* nnz_C,
                        const alphasparse_mat_info_t info_C,
                        void* temp_buffer);

alphasparseStatus_t
alphasparse_s_csrgemm(alphasparseHandle_t handle,
                      alphasparseOperation_t trans_A,
                      alphasparseOperation_t trans_B,
                      int m,
                      int n,
                      int k,
                      const float* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const float* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const float* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const float* beta,
                      const alpha_matrix_descr_t descr_D,
                      int nnz_D,
                      const float* csr_val_D,
                      const int* csr_row_ptr_D,
                      const int* csr_col_ind_D,
                      const alpha_matrix_descr_t descr_C,
                      float* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C,
                      const alphasparse_mat_info_t info_C,
                      void* temp_buffer);

alphasparseStatus_t
alphasparse_d_csrgemm(alphasparseHandle_t handle,
                      alphasparseOperation_t trans_A,
                      alphasparseOperation_t trans_B,
                      int m,
                      int n,
                      int k,
                      const double* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const double* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const double* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const double* beta,
                      const alpha_matrix_descr_t descr_D,
                      int nnz_D,
                      const double* csr_val_D,
                      const int* csr_row_ptr_D,
                      const int* csr_col_ind_D,
                      const alpha_matrix_descr_t descr_C,
                      double* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C,
                      const alphasparse_mat_info_t info_C,
                      void* temp_buffer);

alphasparseStatus_t
alphasparse_c_csrgemm(alphasparseHandle_t handle,
                      alphasparseOperation_t trans_A,
                      alphasparseOperation_t trans_B,
                      int m,
                      int n,
                      int k,
                      const cuFloatComplex* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const cuFloatComplex* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const cuFloatComplex* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const cuFloatComplex* beta,
                      const alpha_matrix_descr_t descr_D,
                      int nnz_D,
                      const cuFloatComplex* csr_val_D,
                      const int* csr_row_ptr_D,
                      const int* csr_col_ind_D,
                      const alpha_matrix_descr_t descr_C,
                      cuFloatComplex* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C,
                      const alphasparse_mat_info_t info_C,
                      void* temp_buffer);

alphasparseStatus_t
alphasparse_z_csrgemm(alphasparseHandle_t handle,
                      alphasparseOperation_t trans_A,
                      alphasparseOperation_t trans_B,
                      int m,
                      int n,
                      int k,
                      const cuDoubleComplex* alpha,
                      const alpha_matrix_descr_t descr_A,
                      int nnz_A,
                      const cuDoubleComplex* csr_val_A,
                      const int* csr_row_ptr_A,
                      const int* csr_col_ind_A,
                      const alpha_matrix_descr_t descr_B,
                      int nnz_B,
                      const cuDoubleComplex* csr_val_B,
                      const int* csr_row_ptr_B,
                      const int* csr_col_ind_B,
                      const cuDoubleComplex* beta,
                      const alpha_matrix_descr_t descr_D,
                      int nnz_D,
                      const cuDoubleComplex* csr_val_D,
                      const int* csr_row_ptr_D,
                      const int* csr_col_ind_D,
                      const alpha_matrix_descr_t descr_C,
                      cuDoubleComplex* csr_val_C,
                      const int* csr_row_ptr_C,
                      int* csr_col_ind_C,
                      const alphasparse_mat_info_t info_C,
                      void* temp_buffer);

alphasparseStatus_t
alphasparse_bsric0_zero_pivot(alphasparseHandle_t handle,
                              alphasparse_mat_info_t info,
                              int* position);

alphasparseStatus_t
alphasparse_sbsric0_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const float* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int block_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_dbsric0_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const double* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int block_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_cbsric0_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const cuFloatComplex* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int block_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_zbsric0_buffer_size(alphasparseHandle_t handle,
                                alphasparse_layout_t dir,
                                int mb,
                                int nnzb,
                                const alpha_matrix_descr_t descr,
                                const cuDoubleComplex* bsr_val,
                                const int* bsr_row_ptr,
                                const int* bsr_col_ind,
                                int block_dim,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_sbsric0_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             int mb,
                             int nnzb,
                             const alpha_matrix_descr_t descr,
                             const float* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int block_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_dbsric0_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             int mb,
                             int nnzb,
                             const alpha_matrix_descr_t descr,
                             const double* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int block_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_cbsric0_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             int mb,
                             int nnzb,
                             const alpha_matrix_descr_t descr,
                             const cuFloatComplex* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int block_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_zbsric0_analysis(alphasparseHandle_t handle,
                             alphasparse_layout_t dir,
                             int mb,
                             int nnzb,
                             const alpha_matrix_descr_t descr,
                             const cuDoubleComplex* bsr_val,
                             const int* bsr_row_ptr,
                             const int* bsr_col_ind,
                             int block_dim,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_bsric0_clear(alphasparseHandle_t handle,
                         alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_sbsric0(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    int mb,
                    int nnzb,
                    const alpha_matrix_descr_t descr,
                    float* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int block_dim,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_dbsric0(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    int mb,
                    int nnzb,
                    const alpha_matrix_descr_t descr,
                    double* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int block_dim,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_cbsric0(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    int mb,
                    int nnzb,
                    const alpha_matrix_descr_t descr,
                    cuFloatComplex* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int block_dim,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_zbsric0(alphasparseHandle_t handle,
                    alphasparse_layout_t dir,
                    int mb,
                    int nnzb,
                    const alpha_matrix_descr_t descr,
                    cuDoubleComplex* bsr_val,
                    const int* bsr_row_ptr,
                    const int* bsr_col_ind,
                    int block_dim,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_bsrilu0_zero_pivot(alphasparseHandle_t handle,
                               alphasparse_mat_info_t info,
                               int* position);

alphasparseStatus_t
alphasparse_sbsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const float* boost_tol,
                                   const float* boost_val);

alphasparseStatus_t
alphasparse_dbsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const double* boost_tol,
                                   const double* boost_val);

alphasparseStatus_t
alphasparse_cbsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const float* boost_tol,
                                   const cuFloatComplex* boost_val);

alphasparseStatus_t
alphasparse_zbsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const double* boost_tol,
                                   const cuDoubleComplex* boost_val);

alphasparseStatus_t
alphasparse_sbsrilu0_buffer_size(alphasparseHandle_t handle,
                                 alphasparse_layout_t dir,
                                 int mb,
                                 int nnzb,
                                 const alpha_matrix_descr_t descr,
                                 const float* bsr_val,
                                 const int* bsr_row_ptr,
                                 const int* bsr_col_ind,
                                 int block_dim,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_dbsrilu0_buffer_size(alphasparseHandle_t handle,
                                 alphasparse_layout_t dir,
                                 int mb,
                                 int nnzb,
                                 const alpha_matrix_descr_t descr,
                                 const double* bsr_val,
                                 const int* bsr_row_ptr,
                                 const int* bsr_col_ind,
                                 int block_dim,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_cbsrilu0_buffer_size(alphasparseHandle_t handle,
                                 alphasparse_layout_t dir,
                                 int mb,
                                 int nnzb,
                                 const alpha_matrix_descr_t descr,
                                 const cuFloatComplex* bsr_val,
                                 const int* bsr_row_ptr,
                                 const int* bsr_col_ind,
                                 int block_dim,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_zbsrilu0_buffer_size(alphasparseHandle_t handle,
                                 alphasparse_layout_t dir,
                                 int mb,
                                 int nnzb,
                                 const alpha_matrix_descr_t descr,
                                 const cuDoubleComplex* bsr_val,
                                 const int* bsr_row_ptr,
                                 const int* bsr_col_ind,
                                 int block_dim,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_sbsrilu0_analysis(alphasparseHandle_t handle,
                              alphasparse_layout_t dir,
                              int mb,
                              int nnzb,
                              const alpha_matrix_descr_t descr,
                              const float* bsr_val,
                              const int* bsr_row_ptr,
                              const int* bsr_col_ind,
                              int block_dim,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_dbsrilu0_analysis(alphasparseHandle_t handle,
                              alphasparse_layout_t dir,
                              int mb,
                              int nnzb,
                              const alpha_matrix_descr_t descr,
                              const double* bsr_val,
                              const int* bsr_row_ptr,
                              const int* bsr_col_ind,
                              int block_dim,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_cbsrilu0_analysis(alphasparseHandle_t handle,
                              alphasparse_layout_t dir,
                              int mb,
                              int nnzb,
                              const alpha_matrix_descr_t descr,
                              const cuFloatComplex* bsr_val,
                              const int* bsr_row_ptr,
                              const int* bsr_col_ind,
                              int block_dim,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_zbsrilu0_analysis(alphasparseHandle_t handle,
                              alphasparse_layout_t dir,
                              int mb,
                              int nnzb,
                              const alpha_matrix_descr_t descr,
                              const cuDoubleComplex* bsr_val,
                              const int* bsr_row_ptr,
                              const int* bsr_col_ind,
                              int block_dim,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_bsrilu0_clear(alphasparseHandle_t handle,
                          alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_sbsrilu0(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nnzb,
                     const alpha_matrix_descr_t descr,
                     float* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_dbsrilu0(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nnzb,
                     const alpha_matrix_descr_t descr,
                     double* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_cbsrilu0(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nnzb,
                     const alpha_matrix_descr_t descr,
                     cuFloatComplex* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_zbsrilu0(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nnzb,
                     const alpha_matrix_descr_t descr,
                     cuDoubleComplex* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_csric0_zero_pivot(alphasparseHandle_t handle,
                              alphasparse_mat_info_t info,
                              int* position);

alphasparseStatus_t
alphasparse_scsric0_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const float* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_dcsric0_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const double* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_ccsric0_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const cuFloatComplex* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_zcsric0_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alpha_matrix_descr_t descr,
                                const cuDoubleComplex* csr_val,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_mat_info_t info,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_scsric0_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const float* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_dcsric0_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const double* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_ccsric0_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const cuFloatComplex* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_zcsric0_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alpha_matrix_descr_t descr,
                             const cuDoubleComplex* csr_val,
                             const int* csr_row_ptr,
                             const int* csr_col_ind,
                             alphasparse_mat_info_t info,
                             alphasparse_analysis_policy_t analysis,
                             alphasparse_solve_policy_t solve,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_csric0_clear(alphasparseHandle_t handle,
                         alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_scsric0(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alpha_matrix_descr_t descr,
                    float* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_dcsric0(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alpha_matrix_descr_t descr,
                    double* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_ccsric0(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alpha_matrix_descr_t descr,
                    cuFloatComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_zcsric0(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alpha_matrix_descr_t descr,
                    cuDoubleComplex* csr_val,
                    const int* csr_row_ptr,
                    const int* csr_col_ind,
                    alphasparse_mat_info_t info,
                    alphasparse_solve_policy_t policy,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_csrilu0_zero_pivot(alphasparseHandle_t handle,
                               alphasparse_mat_info_t info,
                               int* position);

alphasparseStatus_t
alphasparse_scsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const float* boost_tol,
                                   const float* boost_val);

alphasparseStatus_t
alphasparse_dcsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const double* boost_tol,
                                   const double* boost_val);

alphasparseStatus_t
alphasparse_ccsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const float* boost_tol,
                                   const cuFloatComplex* boost_val);

alphasparseStatus_t
alphasparse_zcsrilu0_numeric_boost(alphasparseHandle_t handle,
                                   alphasparse_mat_info_t info,
                                   int enable_boost,
                                   const double* boost_tol,
                                   const cuDoubleComplex* boost_val);

alphasparseStatus_t
alphasparse_scsrilu0_buffer_size(alphasparseHandle_t handle,
                                 int m,
                                 int nnz,
                                 const alpha_matrix_descr_t descr,
                                 const float* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_dcsrilu0_buffer_size(alphasparseHandle_t handle,
                                 int m,
                                 int nnz,
                                 const alpha_matrix_descr_t descr,
                                 const double* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_ccsrilu0_buffer_size(alphasparseHandle_t handle,
                                 int m,
                                 int nnz,
                                 const alpha_matrix_descr_t descr,
                                 const cuFloatComplex* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_zcsrilu0_buffer_size(alphasparseHandle_t handle,
                                 int m,
                                 int nnz,
                                 const alpha_matrix_descr_t descr,
                                 const cuDoubleComplex* csr_val,
                                 const int* csr_row_ptr,
                                 const int* csr_col_ind,
                                 alphasparse_mat_info_t info,
                                 size_t* buffer_size);

alphasparseStatus_t
alphasparse_scsrilu0_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alpha_matrix_descr_t descr,
                              const float* csr_val,
                              const int* csr_row_ptr,
                              const int* csr_col_ind,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_dcsrilu0_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alpha_matrix_descr_t descr,
                              const double* csr_val,
                              const int* csr_row_ptr,
                              const int* csr_col_ind,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_ccsrilu0_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alpha_matrix_descr_t descr,
                              const cuFloatComplex* csr_val,
                              const int* csr_row_ptr,
                              const int* csr_col_ind,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_zcsrilu0_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alpha_matrix_descr_t descr,
                              const cuDoubleComplex* csr_val,
                              const int* csr_row_ptr,
                              const int* csr_col_ind,
                              alphasparse_mat_info_t info,
                              alphasparse_analysis_policy_t analysis,
                              alphasparse_solve_policy_t solve,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_csrilu0_clear(alphasparseHandle_t handle,
                          alphasparse_mat_info_t info);

alphasparseStatus_t
alphasparse_scsrilu0(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alpha_matrix_descr_t descr,
                     float* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_dcsrilu0(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alpha_matrix_descr_t descr,
                     double* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_ccsrilu0(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alpha_matrix_descr_t descr,
                     cuFloatComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_zcsrilu0(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alpha_matrix_descr_t descr,
                     cuDoubleComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_mat_info_t info,
                     alphasparse_solve_policy_t policy,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_snnz(alphasparseHandle_t handle,
                 alphasparse_layout_t dir,
                 int m,
                 int n,
                 const alpha_matrix_descr_t descr,
                 const float* A,
                 int ld,
                 int* nnz_per_row_columns,
                 int* nnz_total_dev_host_ptr);

alphasparseStatus_t
alphasparse_dnnz(alphasparseHandle_t handle,
                 alphasparse_layout_t dir,
                 int m,
                 int n,
                 const alpha_matrix_descr_t descr,
                 const double* A,
                 int ld,
                 int* nnz_per_row_columns,
                 int* nnz_total_dev_host_ptr);

alphasparseStatus_t
alphasparse_cnnz(alphasparseHandle_t handle,
                 alphasparse_layout_t dir,
                 int m,
                 int n,
                 const alpha_matrix_descr_t descr,
                 const cuFloatComplex* A,
                 int ld,
                 int* nnz_per_row_columns,
                 int* nnz_total_dev_host_ptr);

alphasparseStatus_t
alphasparse_znnz(alphasparseHandle_t handle,
                 alphasparse_layout_t dir,
                 int m,
                 int n,
                 const alpha_matrix_descr_t descr,
                 const cuDoubleComplex* A,
                 int ld,
                 int* nnz_per_row_columns,
                 int* nnz_total_dev_host_ptr);

alphasparseStatus_t
alphasparse_sdense2csr(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const float* A,
                       int ld,
                       const int* nnz_per_rows,
                       float* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_ddense2csr(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const double* A,
                       int ld,
                       const int* nnz_per_rows,
                       double* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_cdense2csr(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuFloatComplex* A,
                       int ld,
                       const int* nnz_per_rows,
                       cuFloatComplex* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_zdense2csr(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuDoubleComplex* A,
                       int ld,
                       const int* nnz_per_rows,
                       cuDoubleComplex* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_sprune_dense2csr_buffer_size(alphasparseHandle_t handle,
                                         int m,
                                         int n,
                                         const float* A,
                                         int lda,
                                         const float* threshold,
                                         const alpha_matrix_descr_t descr,
                                         const float* csr_val,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         size_t* buffer_size);

alphasparseStatus_t
alphasparse_dprune_dense2csr_buffer_size(alphasparseHandle_t handle,
                                         int m,
                                         int n,
                                         const double* A,
                                         int lda,
                                         const double* threshold,
                                         const alpha_matrix_descr_t descr,
                                         const double* csr_val,
                                         const int* csr_row_ptr,
                                         const int* csr_col_ind,
                                         size_t* buffer_size);

alphasparseStatus_t
alphasparse_sprune_dense2csr_nnz(alphasparseHandle_t handle,
                                 int m,
                                 int n,
                                 const float* A,
                                 int lda,
                                 const float* threshold,
                                 const alpha_matrix_descr_t descr,
                                 int* csr_row_ptr,
                                 int* nnz_total_dev_host_ptr,
                                 void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_dense2csr_nnz(alphasparseHandle_t handle,
                                 int m,
                                 int n,
                                 const double* A,
                                 int lda,
                                 const double* threshold,
                                 const alpha_matrix_descr_t descr,
                                 int* csr_row_ptr,
                                 int* nnz_total_dev_host_ptr,
                                 void* temp_buffer);

alphasparseStatus_t
alphasparse_sprune_dense2csr(alphasparseHandle_t handle,
                             int m,
                             int n,
                             const float* A,
                             int lda,
                             const float* threshold,
                             const alpha_matrix_descr_t descr,
                             float* csr_val,
                             const int* csr_row_ptr,
                             int* csr_col_ind,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_dense2csr(alphasparseHandle_t handle,
                             int m,
                             int n,
                             const double* A,
                             int lda,
                             const double* threshold,
                             const alpha_matrix_descr_t descr,
                             double* csr_val,
                             const int* csr_row_ptr,
                             int* csr_col_ind,
                             void* temp_buffer);

alphasparseStatus_t
alphasparse_sprune_dense2csr_by_percentage_buffer_size(
  alphasparseHandle_t handle,
  int m,
  int n,
  const float* A,
  int lda,
  float percentage,
  const alpha_matrix_descr_t descr,
  const float* csr_val,
  const int* csr_row_ptr,
  const int* csr_col_ind,
  alphasparse_mat_info_t info,
  size_t* buffer_size);

alphasparseStatus_t
alphasparse_dprune_dense2csr_by_percentage_buffer_size(
  alphasparseHandle_t handle,
  int m,
  int n,
  const double* A,
  int lda,
  double percentage,
  const alpha_matrix_descr_t descr,
  const double* csr_val,
  const int* csr_row_ptr,
  const int* csr_col_ind,
  alphasparse_mat_info_t info,
  size_t* buffer_size);

alphasparseStatus_t
alphasparse_sprune_dense2csr_nnz_by_percentage(alphasparseHandle_t handle,
                                               int m,
                                               int n,
                                               const float* A,
                                               int lda,
                                               float percentage,
                                               const alpha_matrix_descr_t descr,
                                               int* csr_row_ptr,
                                               int* nnz_total_dev_host_ptr,
                                               alphasparse_mat_info_t info,
                                               void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_dense2csr_nnz_by_percentage(alphasparseHandle_t handle,
                                               int m,
                                               int n,
                                               const double* A,
                                               int lda,
                                               double percentage,
                                               const alpha_matrix_descr_t descr,
                                               int* csr_row_ptr,
                                               int* nnz_total_dev_host_ptr,
                                               alphasparse_mat_info_t info,
                                               void* temp_buffer);

alphasparseStatus_t
alphasparse_sprune_dense2csr_by_percentage(alphasparseHandle_t handle,
                                           int m,
                                           int n,
                                           const float* A,
                                           int lda,
                                           float percentage,
                                           const alpha_matrix_descr_t descr,
                                           float* csr_val,
                                           const int* csr_row_ptr,
                                           int* csr_col_ind,
                                           alphasparse_mat_info_t info,
                                           void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_dense2csr_by_percentage(alphasparseHandle_t handle,
                                           int m,
                                           int n,
                                           const double* A,
                                           int lda,
                                           double percentage,
                                           const alpha_matrix_descr_t descr,
                                           double* csr_val,
                                           const int* csr_row_ptr,
                                           int* csr_col_ind,
                                           alphasparse_mat_info_t info,
                                           void* temp_buffer);

alphasparseStatus_t
alphasparse_sdense2csc(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const float* A,
                       int ld,
                       const int* nnz_per_columns,
                       float* csc_val,
                       int* csc_col_ptr,
                       int* csc_row_ind);

alphasparseStatus_t
alphasparse_ddense2csc(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const double* A,
                       int ld,
                       const int* nnz_per_columns,
                       double* csc_val,
                       int* csc_col_ptr,
                       int* csc_row_ind);

alphasparseStatus_t
alphasparse_cdense2csc(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuFloatComplex* A,
                       int ld,
                       const int* nnz_per_columns,
                       cuFloatComplex* csc_val,
                       int* csc_col_ptr,
                       int* csc_row_ind);

alphasparseStatus_t
alphasparse_zdense2csc(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuDoubleComplex* A,
                       int ld,
                       const int* nnz_per_columns,
                       cuDoubleComplex* csc_val,
                       int* csc_col_ptr,
                       int* csc_row_ind);

alphasparseStatus_t
alphasparse_sdense2coo(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const float* A,
                       int ld,
                       const int* nnz_per_rows,
                       float* coo_val,
                       int* coo_row_ind,
                       int* coo_col_ind);

alphasparseStatus_t
alphasparse_ddense2coo(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const double* A,
                       int ld,
                       const int* nnz_per_rows,
                       double* coo_val,
                       int* coo_row_ind,
                       int* coo_col_ind);

alphasparseStatus_t
alphasparse_cdense2coo(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuFloatComplex* A,
                       int ld,
                       const int* nnz_per_rows,
                       cuFloatComplex* coo_val,
                       int* coo_row_ind,
                       int* coo_col_ind);

alphasparseStatus_t
alphasparse_zdense2coo(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuDoubleComplex* A,
                       int ld,
                       const int* nnz_per_rows,
                       cuDoubleComplex* coo_val,
                       int* coo_row_ind,
                       int* coo_col_ind);

alphasparseStatus_t
alphasparse_scsr2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const float* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       float* A,
                       int ld);

alphasparseStatus_t
alphasparse_dcsr2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const double* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       double* A,
                       int ld);

alphasparseStatus_t
alphasparse_ccsr2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuFloatComplex* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       cuFloatComplex* A,
                       int ld);

alphasparseStatus_t
alphasparse_zcsr2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuDoubleComplex* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       cuDoubleComplex* A,
                       int ld);

alphasparseStatus_t
alphasparse_scsc2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const float* csc_val,
                       const int* csc_col_ptr,
                       const int* csc_row_ind,
                       float* A,
                       int ld);

alphasparseStatus_t
alphasparse_dcsc2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const double* csc_val,
                       const int* csc_col_ptr,
                       const int* csc_row_ind,
                       double* A,
                       int ld);

alphasparseStatus_t
alphasparse_ccsc2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuFloatComplex* csc_val,
                       const int* csc_col_ptr,
                       const int* csc_row_ind,
                       cuFloatComplex* A,
                       int ld);

alphasparseStatus_t
alphasparse_zcsc2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       const alpha_matrix_descr_t descr,
                       const cuDoubleComplex* csc_val,
                       const int* csc_col_ptr,
                       const int* csc_row_ind,
                       cuDoubleComplex* A,
                       int ld);

alphasparseStatus_t
alphasparse_scoo2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       int nnz,
                       const alpha_matrix_descr_t descr,
                       const float* coo_val,
                       const int* coo_row_ind,
                       const int* coo_col_ind,
                       float* A,
                       int ld);

alphasparseStatus_t
alphasparse_dcoo2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       int nnz,
                       const alpha_matrix_descr_t descr,
                       const double* coo_val,
                       const int* coo_row_ind,
                       const int* coo_col_ind,
                       double* A,
                       int ld);

alphasparseStatus_t
alphasparse_ccoo2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       int nnz,
                       const alpha_matrix_descr_t descr,
                       const cuFloatComplex* coo_val,
                       const int* coo_row_ind,
                       const int* coo_col_ind,
                       cuFloatComplex* A,
                       int ld);

alphasparseStatus_t
alphasparse_zcoo2dense(alphasparseHandle_t handle,
                       int m,
                       int n,
                       int nnz,
                       const alpha_matrix_descr_t descr,
                       const cuDoubleComplex* coo_val,
                       const int* coo_row_ind,
                       const int* coo_col_ind,
                       cuDoubleComplex* A,
                       int ld);

alphasparseStatus_t
alphasparse_snnz_compress(alphasparseHandle_t handle,
                          int m,
                          const alpha_matrix_descr_t descr_A,
                          const float* csr_val_A,
                          const int* csr_row_ptr_A,
                          int* nnz_per_row,
                          int* nnz_C,
                          float tol);

alphasparseStatus_t
alphasparse_dnnz_compress(alphasparseHandle_t handle,
                          int m,
                          const alpha_matrix_descr_t descr_A,
                          const double* csr_val_A,
                          const int* csr_row_ptr_A,
                          int* nnz_per_row,
                          int* nnz_C,
                          double tol);

alphasparseStatus_t
alphasparse_cnnz_compress(alphasparseHandle_t handle,
                          int m,
                          const alpha_matrix_descr_t descr_A,
                          const cuFloatComplex* csr_val_A,
                          const int* csr_row_ptr_A,
                          int* nnz_per_row,
                          int* nnz_C,
                          cuFloatComplex tol);

alphasparseStatus_t
alphasparse_znnz_compress(alphasparseHandle_t handle,
                          int m,
                          const alpha_matrix_descr_t descr_A,
                          const cuDoubleComplex* csr_val_A,
                          const int* csr_row_ptr_A,
                          int* nnz_per_row,
                          int* nnz_C,
                          cuDoubleComplex tol);

alphasparseStatus_t
alphasparse_csr2coo(alphasparseHandle_t handle,
                    const int* csr_row_ptr,
                    int nnz,
                    int m,
                    int* coo_row_ind,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_csr2csc_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                alphasparse_action_t copy_values,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_scsr2csc(alphasparseHandle_t handle,
                     int m,
                     int n,
                     int nnz,
                     const float* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     float* csc_val,
                     int* csc_row_ind,
                     int* csc_col_ptr,
                     alphasparse_action_t copy_values,
                     alphasparseIndexBase_t idx_base,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_dcsr2csc(alphasparseHandle_t handle,
                     int m,
                     int n,
                     int nnz,
                     const double* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     double* csc_val,
                     int* csc_row_ind,
                     int* csc_col_ptr,
                     alphasparse_action_t copy_values,
                     alphasparseIndexBase_t idx_base,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_ccsr2csc(alphasparseHandle_t handle,
                     int m,
                     int n,
                     int nnz,
                     const cuFloatComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     cuFloatComplex* csc_val,
                     int* csc_row_ind,
                     int* csc_col_ptr,
                     alphasparse_action_t copy_values,
                     alphasparseIndexBase_t idx_base,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_zcsr2csc(alphasparseHandle_t handle,
                     int m,
                     int n,
                     int nnz,
                     const cuDoubleComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     cuDoubleComplex* csc_val,
                     int* csc_row_ind,
                     int* csc_col_ptr,
                     alphasparse_action_t copy_values,
                     alphasparseIndexBase_t idx_base,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_sgebsr2gebsc_buffer_size(alphasparseHandle_t handle,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const float* bsr_val,
                                     const int* bsr_row_ptr,
                                     const int* bsr_col_ind,
                                     int row_block_dim,
                                     int col_block_dim,
                                     size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_dgebsr2gebsc_buffer_size(alphasparseHandle_t handle,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const double* bsr_val,
                                     const int* bsr_row_ptr,
                                     const int* bsr_col_ind,
                                     int row_block_dim,
                                     int col_block_dim,
                                     size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_cgebsr2gebsc_buffer_size(alphasparseHandle_t handle,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const cuFloatComplex* bsr_val,
                                     const int* bsr_row_ptr,
                                     const int* bsr_col_ind,
                                     int row_block_dim,
                                     int col_block_dim,
                                     size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_zgebsr2gebsc_buffer_size(alphasparseHandle_t handle,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const cuDoubleComplex* bsr_val,
                                     const int* bsr_row_ptr,
                                     const int* bsr_col_ind,
                                     int row_block_dim,
                                     int col_block_dim,
                                     size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_sgebsr2gebsc(alphasparseHandle_t handle,
                         int mb,
                         int nb,
                         int nnzb,
                         const float* bsr_val,
                         const int* bsr_row_ptr,
                         const int* bsr_col_ind,
                         int row_block_dim,
                         int col_block_dim,
                         float* bsc_val,
                         int* bsc_row_ind,
                         int* bsc_col_ptr,
                         alphasparse_action_t copy_values,
                         alphasparseIndexBase_t idx_base,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_dgebsr2gebsc(alphasparseHandle_t handle,
                         int mb,
                         int nb,
                         int nnzb,
                         const double* bsr_val,
                         const int* bsr_row_ptr,
                         const int* bsr_col_ind,
                         int row_block_dim,
                         int col_block_dim,
                         double* bsc_val,
                         int* bsc_row_ind,
                         int* bsc_col_ptr,
                         alphasparse_action_t copy_values,
                         alphasparseIndexBase_t idx_base,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_cgebsr2gebsc(alphasparseHandle_t handle,
                         int mb,
                         int nb,
                         int nnzb,
                         const cuFloatComplex* bsr_val,
                         const int* bsr_row_ptr,
                         const int* bsr_col_ind,
                         int row_block_dim,
                         int col_block_dim,
                         cuFloatComplex* bsc_val,
                         int* bsc_row_ind,
                         int* bsc_col_ptr,
                         alphasparse_action_t copy_values,
                         alphasparseIndexBase_t idx_base,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_zgebsr2gebsc(alphasparseHandle_t handle,
                         int mb,
                         int nb,
                         int nnzb,
                         const cuDoubleComplex* bsr_val,
                         const int* bsr_row_ptr,
                         const int* bsr_col_ind,
                         int row_block_dim,
                         int col_block_dim,
                         cuDoubleComplex* bsc_val,
                         int* bsc_row_ind,
                         int* bsc_col_ptr,
                         alphasparse_action_t copy_values,
                         alphasparseIndexBase_t idx_base,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_csr2ell_width(alphasparseHandle_t handle,
                          int m,
                          const alpha_matrix_descr_t csr_descr,
                          const int* csr_row_ptr,
                          const alpha_matrix_descr_t ell_descr,
                          int* ell_width);

alphasparseStatus_t
alphasparse_scsr2ell(alphasparseHandle_t handle,
                     int m,
                     const alpha_matrix_descr_t csr_descr,
                     const float* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     float* ell_val,
                     int* ell_col_ind);

alphasparseStatus_t
alphasparse_dcsr2ell(alphasparseHandle_t handle,
                     int m,
                     const alpha_matrix_descr_t csr_descr,
                     const double* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     double* ell_val,
                     int* ell_col_ind);

alphasparseStatus_t
alphasparse_ccsr2ell(alphasparseHandle_t handle,
                     int m,
                     const alpha_matrix_descr_t csr_descr,
                     const cuFloatComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     cuFloatComplex* ell_val,
                     int* ell_col_ind);

alphasparseStatus_t
alphasparse_zcsr2ell(alphasparseHandle_t handle,
                     int m,
                     const alpha_matrix_descr_t csr_descr,
                     const cuDoubleComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     cuDoubleComplex* ell_val,
                     int* ell_col_ind);

alphasparseStatus_t
alphasparse_scsr2hyb(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t descr,
                     const float* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_hyb_mat_t hyb,
                     int user_ell_width,
                     alphasparse_hyb_partition_t partition_type);

alphasparseStatus_t
alphasparse_dcsr2hyb(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t descr,
                     const double* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_hyb_mat_t hyb,
                     int user_ell_width,
                     alphasparse_hyb_partition_t partition_type);

alphasparseStatus_t
alphasparse_ccsr2hyb(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t descr,
                     const cuFloatComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_hyb_mat_t hyb,
                     int user_ell_width,
                     alphasparse_hyb_partition_t partition_type);

alphasparseStatus_t
alphasparse_zcsr2hyb(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t descr,
                     const cuDoubleComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     alphasparse_hyb_mat_t hyb,
                     int user_ell_width,
                     alphasparse_hyb_partition_t partition_type);

alphasparseStatus_t
alphasparse_csr2bsr_nnz(alphasparseHandle_t handle,
                        alphasparse_layout_t dir,
                        int m,
                        int n,
                        const alpha_matrix_descr_t csr_descr,
                        const int* csr_row_ptr,
                        const int* csr_col_ind,
                        int block_dim,
                        const alpha_matrix_descr_t bsr_descr,
                        int* bsr_row_ptr,
                        int* bsr_nnz);

alphasparseStatus_t
alphasparse_scsr2bsr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int m,
                     int n,
                     const alpha_matrix_descr_t csr_descr,
                     const float* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t bsr_descr,
                     float* bsr_val,
                     int* bsr_row_ptr,
                     int* bsr_col_ind);

alphasparseStatus_t
alphasparse_dcsr2bsr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int m,
                     int n,
                     const alpha_matrix_descr_t csr_descr,
                     const double* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t bsr_descr,
                     double* bsr_val,
                     int* bsr_row_ptr,
                     int* bsr_col_ind);

alphasparseStatus_t
alphasparse_ccsr2bsr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int m,
                     int n,
                     const alpha_matrix_descr_t csr_descr,
                     const cuFloatComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t bsr_descr,
                     cuFloatComplex* bsr_val,
                     int* bsr_row_ptr,
                     int* bsr_col_ind);

alphasparseStatus_t
alphasparse_zcsr2bsr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int m,
                     int n,
                     const alpha_matrix_descr_t csr_descr,
                     const cuDoubleComplex* csr_val,
                     const int* csr_row_ptr,
                     const int* csr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t bsr_descr,
                     cuDoubleComplex* bsr_val,
                     int* bsr_row_ptr,
                     int* bsr_col_ind);

alphasparseStatus_t
alphasparse_scsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                   alphasparse_layout_t dir,
                                   int m,
                                   int n,
                                   const alpha_matrix_descr_t csr_descr,
                                   const float* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   int row_block_dim,
                                   int col_block_dim,
                                   size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_dcsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                   alphasparse_layout_t dir,
                                   int m,
                                   int n,
                                   const alpha_matrix_descr_t csr_descr,
                                   const double* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   int row_block_dim,
                                   int col_block_dim,
                                   size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_ccsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                   alphasparse_layout_t dir,
                                   int m,
                                   int n,
                                   const alpha_matrix_descr_t csr_descr,
                                   const cuFloatComplex* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   int row_block_dim,
                                   int col_block_dim,
                                   size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_zcsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                   alphasparse_layout_t dir,
                                   int m,
                                   int n,
                                   const alpha_matrix_descr_t csr_descr,
                                   const cuDoubleComplex* csr_val,
                                   const int* csr_row_ptr,
                                   const int* csr_col_ind,
                                   int row_block_dim,
                                   int col_block_dim,
                                   size_t* p_buffer_size);

alphasparseStatus_t
alphasparse_csr2gebsr_nnz(alphasparseHandle_t handle,
                          alphasparse_layout_t dir,
                          int m,
                          int n,
                          const alpha_matrix_descr_t csr_descr,
                          const int* csr_row_ptr,
                          const int* csr_col_ind,
                          const alpha_matrix_descr_t bsr_descr,
                          int* bsr_row_ptr,
                          int row_block_dim,
                          int col_block_dim,
                          int* bsr_nnz_devhost,
                          void* p_buffer);

alphasparseStatus_t
alphasparse_scsr2gebsr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int m,
                       int n,
                       const alpha_matrix_descr_t csr_descr,
                       const float* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       const alpha_matrix_descr_t bsr_descr,
                       float* bsr_val,
                       int* bsr_row_ptr,
                       int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       void* p_buffer);

alphasparseStatus_t
alphasparse_dcsr2gebsr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int m,
                       int n,
                       const alpha_matrix_descr_t csr_descr,
                       const double* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       const alpha_matrix_descr_t bsr_descr,
                       double* bsr_val,
                       int* bsr_row_ptr,
                       int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       void* p_buffer);

alphasparseStatus_t
alphasparse_ccsr2gebsr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int m,
                       int n,
                       const alpha_matrix_descr_t csr_descr,
                       const cuFloatComplex* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       const alpha_matrix_descr_t bsr_descr,
                       cuFloatComplex* bsr_val,
                       int* bsr_row_ptr,
                       int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       void* p_buffer);

alphasparseStatus_t
alphasparse_zcsr2gebsr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int m,
                       int n,
                       const alpha_matrix_descr_t csr_descr,
                       const cuDoubleComplex* csr_val,
                       const int* csr_row_ptr,
                       const int* csr_col_ind,
                       const alpha_matrix_descr_t bsr_descr,
                       cuDoubleComplex* bsr_val,
                       int* bsr_row_ptr,
                       int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       void* p_buffer);

alphasparseStatus_t
alphasparse_scsr2csr_compress(alphasparseHandle_t handle,
                              int m,
                              int n,
                              const alpha_matrix_descr_t descr_A,
                              const float* csr_val_A,
                              const int* csr_row_ptr_A,
                              const int* csr_col_ind_A,
                              int nnz_A,
                              const int* nnz_per_row,
                              float* csr_val_C,
                              int* csr_row_ptr_C,
                              int* csr_col_ind_C,
                              float tol);

alphasparseStatus_t
alphasparse_dcsr2csr_compress(alphasparseHandle_t handle,
                              int m,
                              int n,
                              const alpha_matrix_descr_t descr_A,
                              const double* csr_val_A,
                              const int* csr_row_ptr_A,
                              const int* csr_col_ind_A,
                              int nnz_A,
                              const int* nnz_per_row,
                              double* csr_val_C,
                              int* csr_row_ptr_C,
                              int* csr_col_ind_C,
                              double tol);

alphasparseStatus_t
alphasparse_ccsr2csr_compress(alphasparseHandle_t handle,
                              int m,
                              int n,
                              const alpha_matrix_descr_t descr_A,
                              const cuFloatComplex* csr_val_A,
                              const int* csr_row_ptr_A,
                              const int* csr_col_ind_A,
                              int nnz_A,
                              const int* nnz_per_row,
                              cuFloatComplex* csr_val_C,
                              int* csr_row_ptr_C,
                              int* csr_col_ind_C,
                              cuFloatComplex tol);

alphasparseStatus_t
alphasparse_zcsr2csr_compress(alphasparseHandle_t handle,
                              int m,
                              int n,
                              const alpha_matrix_descr_t descr_A,
                              const cuDoubleComplex* csr_val_A,
                              const int* csr_row_ptr_A,
                              const int* csr_col_ind_A,
                              int nnz_A,
                              const int* nnz_per_row,
                              cuDoubleComplex* csr_val_C,
                              int* csr_row_ptr_C,
                              int* csr_col_ind_C,
                              cuDoubleComplex tol);

alphasparseStatus_t
alphasparse_sprune_csr2csr_buffer_size(alphasparseHandle_t handle,
                                       int m,
                                       int n,
                                       int nnz_A,
                                       const alpha_matrix_descr_t csr_descr_A,
                                       const float* csr_val_A,
                                       const int* csr_row_ptr_A,
                                       const int* csr_col_ind_A,
                                       const float* threshold,
                                       const alpha_matrix_descr_t csr_descr_C,
                                       const float* csr_val_C,
                                       const int* csr_row_ptr_C,
                                       const int* csr_col_ind_C,
                                       size_t* buffer_size);

alphasparseStatus_t
alphasparse_dprune_csr2csr_buffer_size(alphasparseHandle_t handle,
                                       int m,
                                       int n,
                                       int nnz_A,
                                       const alpha_matrix_descr_t csr_descr_A,
                                       const double* csr_val_A,
                                       const int* csr_row_ptr_A,
                                       const int* csr_col_ind_A,
                                       const double* threshold,
                                       const alpha_matrix_descr_t csr_descr_C,
                                       const double* csr_val_C,
                                       const int* csr_row_ptr_C,
                                       const int* csr_col_ind_C,
                                       size_t* buffer_size);

alphasparseStatus_t
alphasparse_sprune_csr2csr_nnz(alphasparseHandle_t handle,
                               int m,
                               int n,
                               int nnz_A,
                               const alpha_matrix_descr_t csr_descr_A,
                               const float* csr_val_A,
                               const int* csr_row_ptr_A,
                               const int* csr_col_ind_A,
                               const float* threshold,
                               const alpha_matrix_descr_t csr_descr_C,
                               int* csr_row_ptr_C,
                               int* nnz_total_dev_host_ptr,
                               void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_csr2csr_nnz(alphasparseHandle_t handle,
                               int m,
                               int n,
                               int nnz_A,
                               const alpha_matrix_descr_t csr_descr_A,
                               const double* csr_val_A,
                               const int* csr_row_ptr_A,
                               const int* csr_col_ind_A,
                               const double* threshold,
                               const alpha_matrix_descr_t csr_descr_C,
                               int* csr_row_ptr_C,
                               int* nnz_total_dev_host_ptr,
                               void* temp_buffer);

alphasparseStatus_t
alphasparse_sprune_csr2csr(alphasparseHandle_t handle,
                           int m,
                           int n,
                           int nnz_A,
                           const alpha_matrix_descr_t csr_descr_A,
                           const float* csr_val_A,
                           const int* csr_row_ptr_A,
                           const int* csr_col_ind_A,
                           const float* threshold,
                           const alpha_matrix_descr_t csr_descr_C,
                           float* csr_val_C,
                           const int* csr_row_ptr_C,
                           int* csr_col_ind_C,
                           void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_csr2csr(alphasparseHandle_t handle,
                           int m,
                           int n,
                           int nnz_A,
                           const alpha_matrix_descr_t csr_descr_A,
                           const double* csr_val_A,
                           const int* csr_row_ptr_A,
                           const int* csr_col_ind_A,
                           const double* threshold,
                           const alpha_matrix_descr_t csr_descr_C,
                           double* csr_val_C,
                           const int* csr_row_ptr_C,
                           int* csr_col_ind_C,
                           void* temp_buffer);

alphasparseStatus_t
alphasparse_sprune_csr2csr_by_percentage_buffer_size(
  alphasparseHandle_t handle,
  int m,
  int n,
  int nnz_A,
  const alpha_matrix_descr_t csr_descr_A,
  const float* csr_val_A,
  const int* csr_row_ptr_A,
  const int* csr_col_ind_A,
  float percentage,
  const alpha_matrix_descr_t csr_descr_C,
  const float* csr_val_C,
  const int* csr_row_ptr_C,
  const int* csr_col_ind_C,
  alphasparse_mat_info_t info,
  size_t* buffer_size);

alphasparseStatus_t
alphasparse_dprune_csr2csr_by_percentage_buffer_size(
  alphasparseHandle_t handle,
  int m,
  int n,
  int nnz_A,
  const alpha_matrix_descr_t csr_descr_A,
  const double* csr_val_A,
  const int* csr_row_ptr_A,
  const int* csr_col_ind_A,
  double percentage,
  const alpha_matrix_descr_t csr_descr_C,
  const double* csr_val_C,
  const int* csr_row_ptr_C,
  const int* csr_col_ind_C,
  alphasparse_mat_info_t info,
  size_t* buffer_size);

alphasparseStatus_t
alphasparse_sprune_csr2csr_nnz_by_percentage(
  alphasparseHandle_t handle,
  int m,
  int n,
  int nnz_A,
  const alpha_matrix_descr_t csr_descr_A,
  const float* csr_val_A,
  const int* csr_row_ptr_A,
  const int* csr_col_ind_A,
  float percentage,
  const alpha_matrix_descr_t csr_descr_C,
  int* csr_row_ptr_C,
  int* nnz_total_dev_host_ptr,
  alphasparse_mat_info_t info,
  void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_csr2csr_nnz_by_percentage(
  alphasparseHandle_t handle,
  int m,
  int n,
  int nnz_A,
  const alpha_matrix_descr_t csr_descr_A,
  const double* csr_val_A,
  const int* csr_row_ptr_A,
  const int* csr_col_ind_A,
  double percentage,
  const alpha_matrix_descr_t csr_descr_C,
  int* csr_row_ptr_C,
  int* nnz_total_dev_host_ptr,
  alphasparse_mat_info_t info,
  void* temp_buffer);

alphasparseStatus_t
alphasparse_sprune_csr2csr_by_percentage(alphasparseHandle_t handle,
                                         int m,
                                         int n,
                                         int nnz_A,
                                         const alpha_matrix_descr_t csr_descr_A,
                                         const float* csr_val_A,
                                         const int* csr_row_ptr_A,
                                         const int* csr_col_ind_A,
                                         float percentage,
                                         const alpha_matrix_descr_t csr_descr_C,
                                         float* csr_val_C,
                                         const int* csr_row_ptr_C,
                                         int* csr_col_ind_C,
                                         alphasparse_mat_info_t info,
                                         void* temp_buffer);

alphasparseStatus_t
alphasparse_dprune_csr2csr_by_percentage(alphasparseHandle_t handle,
                                         int m,
                                         int n,
                                         int nnz_A,
                                         const alpha_matrix_descr_t csr_descr_A,
                                         const double* csr_val_A,
                                         const int* csr_row_ptr_A,
                                         const int* csr_col_ind_A,
                                         double percentage,
                                         const alpha_matrix_descr_t csr_descr_C,
                                         double* csr_val_C,
                                         const int* csr_row_ptr_C,
                                         int* csr_col_ind_C,
                                         alphasparse_mat_info_t info,
                                         void* temp_buffer);

alphasparseStatus_t
alphasparse_coo2csr(alphasparseHandle_t handle,
                    const int* coo_row_ind,
                    int nnz,
                    int m,
                    int* csr_row_ptr,
                    alphasparseIndexBase_t idx_base);

alphasparseStatus_t
alphasparse_ell2csr_nnz(alphasparseHandle_t handle,
                        int m,
                        int n,
                        const alpha_matrix_descr_t ell_descr,
                        int ell_width,
                        const int* ell_col_ind,
                        const alpha_matrix_descr_t csr_descr,
                        int* csr_row_ptr,
                        int* csr_nnz);

alphasparseStatus_t
alphasparse_sell2csr(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     const float* ell_val,
                     const int* ell_col_ind,
                     const alpha_matrix_descr_t csr_descr,
                     float* csr_val,
                     const int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_dell2csr(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     const double* ell_val,
                     const int* ell_col_ind,
                     const alpha_matrix_descr_t csr_descr,
                     double* csr_val,
                     const int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_cell2csr(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     const cuFloatComplex* ell_val,
                     const int* ell_col_ind,
                     const alpha_matrix_descr_t csr_descr,
                     cuFloatComplex* csr_val,
                     const int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_zell2csr(alphasparseHandle_t handle,
                     int m,
                     int n,
                     const alpha_matrix_descr_t ell_descr,
                     int ell_width,
                     const cuDoubleComplex* ell_val,
                     const int* ell_col_ind,
                     const alpha_matrix_descr_t csr_descr,
                     cuDoubleComplex* csr_val,
                     const int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_hyb2csr_buffer_size(alphasparseHandle_t handle,
                                const alpha_matrix_descr_t descr,
                                const alphasparse_hyb_mat_t hyb,
                                const int* csr_row_ptr,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_shyb2csr(alphasparseHandle_t handle,
                     const alpha_matrix_descr_t descr,
                     const alphasparse_hyb_mat_t hyb,
                     float* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_dhyb2csr(alphasparseHandle_t handle,
                     const alpha_matrix_descr_t descr,
                     const alphasparse_hyb_mat_t hyb,
                     double* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_chyb2csr(alphasparseHandle_t handle,
                     const alpha_matrix_descr_t descr,
                     const alphasparse_hyb_mat_t hyb,
                     cuFloatComplex* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_zhyb2csr(alphasparseHandle_t handle,
                     const alpha_matrix_descr_t descr,
                     const alphasparse_hyb_mat_t hyb,
                     cuDoubleComplex* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind,
                     void* temp_buffer);

alphasparseStatus_t
alphasparse_create_identity_permutation(alphasparseHandle_t handle,
                                        int n,
                                        int* p);

alphasparseStatus_t
alphasparse_csrsort_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                const int* csr_row_ptr,
                                const int* csr_col_ind,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_csrsort(alphasparseHandle_t handle,
                    int m,
                    int n,
                    int nnz,
                    const alpha_matrix_descr_t descr,
                    const int* csr_row_ptr,
                    int* csr_col_ind,
                    int* perm,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_cscsort_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                const int* csc_col_ptr,
                                const int* csc_row_ind,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_cscsort(alphasparseHandle_t handle,
                    int m,
                    int n,
                    int nnz,
                    const alpha_matrix_descr_t descr,
                    const int* csc_col_ptr,
                    int* csc_row_ind,
                    int* perm,
                    void* temp_buffer);

alphasparseStatus_t
alphasparse_coosort_buffer_size(alphasparseHandle_t handle,
                                int m,
                                int n,
                                int nnz,
                                const int* coo_row_ind,
                                const int* coo_col_ind,
                                size_t* buffer_size);

alphasparseStatus_t
alphasparse_coosort_by_row(alphasparseHandle_t handle,
                           int m,
                           int n,
                           int nnz,
                           int* coo_row_ind,
                           int* coo_col_ind,
                           int* perm,
                           void* temp_buffer);

alphasparseStatus_t
alphasparse_coosort_by_column(alphasparseHandle_t handle,
                              int m,
                              int n,
                              int nnz,
                              int* coo_row_ind,
                              int* coo_col_ind,
                              int* perm,
                              void* temp_buffer);

alphasparseStatus_t
alphasparse_sbsr2csr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nb,
                     const alpha_matrix_descr_t bsr_descr,
                     const float* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t csr_descr,
                     float* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_dbsr2csr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nb,
                     const alpha_matrix_descr_t bsr_descr,
                     const double* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t csr_descr,
                     double* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_cbsr2csr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nb,
                     const alpha_matrix_descr_t bsr_descr,
                     const cuFloatComplex* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t csr_descr,
                     cuFloatComplex* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_zbsr2csr(alphasparseHandle_t handle,
                     alphasparse_layout_t dir,
                     int mb,
                     int nb,
                     const alpha_matrix_descr_t bsr_descr,
                     const cuDoubleComplex* bsr_val,
                     const int* bsr_row_ptr,
                     const int* bsr_col_ind,
                     int block_dim,
                     const alpha_matrix_descr_t csr_descr,
                     cuDoubleComplex* csr_val,
                     int* csr_row_ptr,
                     int* csr_col_ind);

alphasparseStatus_t
alphasparse_sgebsr2csr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int mb,
                       int nb,
                       const alpha_matrix_descr_t bsr_descr,
                       const float* bsr_val,
                       const int* bsr_row_ptr,
                       const int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       const alpha_matrix_descr_t csr_descr,
                       float* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_dgebsr2csr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int mb,
                       int nb,
                       const alpha_matrix_descr_t bsr_descr,
                       const double* bsr_val,
                       const int* bsr_row_ptr,
                       const int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       const alpha_matrix_descr_t csr_descr,
                       double* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_cgebsr2csr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int mb,
                       int nb,
                       const alpha_matrix_descr_t bsr_descr,
                       const cuFloatComplex* bsr_val,
                       const int* bsr_row_ptr,
                       const int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       const alpha_matrix_descr_t csr_descr,
                       cuFloatComplex* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_zgebsr2csr(alphasparseHandle_t handle,
                       alphasparse_layout_t dir,
                       int mb,
                       int nb,
                       const alpha_matrix_descr_t bsr_descr,
                       const cuDoubleComplex* bsr_val,
                       const int* bsr_row_ptr,
                       const int* bsr_col_ind,
                       int row_block_dim,
                       int col_block_dim,
                       const alpha_matrix_descr_t csr_descr,
                       cuDoubleComplex* csr_val,
                       int* csr_row_ptr,
                       int* csr_col_ind);

alphasparseStatus_t
alphasparse_sgebsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                     alphasparse_layout_t dir,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const alpha_matrix_descr_t descr_A,
                                     const float* bsr_val_A,
                                     const int* bsr_row_ptr_A,
                                     const int* bsr_col_ind_A,
                                     int row_block_dim_A,
                                     int col_block_dim_A,
                                     int row_block_dim_C,
                                     int col_block_dim_C,
                                     size_t* buffer_size);

alphasparseStatus_t
alphasparse_dgebsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                     alphasparse_layout_t dir,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const alpha_matrix_descr_t descr_A,
                                     const double* bsr_val_A,
                                     const int* bsr_row_ptr_A,
                                     const int* bsr_col_ind_A,
                                     int row_block_dim_A,
                                     int col_block_dim_A,
                                     int row_block_dim_C,
                                     int col_block_dim_C,
                                     size_t* buffer_size);

alphasparseStatus_t
alphasparse_cgebsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                     alphasparse_layout_t dir,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const alpha_matrix_descr_t descr_A,
                                     const cuFloatComplex* bsr_val_A,
                                     const int* bsr_row_ptr_A,
                                     const int* bsr_col_ind_A,
                                     int row_block_dim_A,
                                     int col_block_dim_A,
                                     int row_block_dim_C,
                                     int col_block_dim_C,
                                     size_t* buffer_size);

alphasparseStatus_t
alphasparse_zgebsr2gebsr_buffer_size(alphasparseHandle_t handle,
                                     alphasparse_layout_t dir,
                                     int mb,
                                     int nb,
                                     int nnzb,
                                     const alpha_matrix_descr_t descr_A,
                                     const cuDoubleComplex* bsr_val_A,
                                     const int* bsr_row_ptr_A,
                                     const int* bsr_col_ind_A,
                                     int row_block_dim_A,
                                     int col_block_dim_A,
                                     int row_block_dim_C,
                                     int col_block_dim_C,
                                     size_t* buffer_size);

alphasparseStatus_t
alphasparse_gebsr2gebsr_nnz(alphasparseHandle_t handle,
                            alphasparse_layout_t dir,
                            int mb,
                            int nb,
                            int nnzb,
                            const alpha_matrix_descr_t descr_A,
                            const int* bsr_row_ptr_A,
                            const int* bsr_col_ind_A,
                            int row_block_dim_A,
                            int col_block_dim_A,
                            const alpha_matrix_descr_t descr_C,
                            int* bsr_row_ptr_C,
                            int row_block_dim_C,
                            int col_block_dim_C,
                            int* nnz_total_dev_host_ptr,
                            void* temp_buffer);

alphasparseStatus_t
alphasparse_sgebsr2gebsr(alphasparseHandle_t handle,
                         alphasparse_layout_t dir,
                         int mb,
                         int nb,
                         int nnzb,
                         const alpha_matrix_descr_t descr_A,
                         const float* bsr_val_A,
                         const int* bsr_row_ptr_A,
                         const int* bsr_col_ind_A,
                         int row_block_dim_A,
                         int col_block_dim_A,
                         const alpha_matrix_descr_t descr_C,
                         float* bsr_val_C,
                         int* bsr_row_ptr_C,
                         int* bsr_col_ind_C,
                         int row_block_dim_C,
                         int col_block_dim_C,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_dgebsr2gebsr(alphasparseHandle_t handle,
                         alphasparse_layout_t dir,
                         int mb,
                         int nb,
                         int nnzb,
                         const alpha_matrix_descr_t descr_A,
                         const double* bsr_val_A,
                         const int* bsr_row_ptr_A,
                         const int* bsr_col_ind_A,
                         int row_block_dim_A,
                         int col_block_dim_A,
                         const alpha_matrix_descr_t descr_C,
                         double* bsr_val_C,
                         int* bsr_row_ptr_C,
                         int* bsr_col_ind_C,
                         int row_block_dim_C,
                         int col_block_dim_C,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_cgebsr2gebsr(alphasparseHandle_t handle,
                         alphasparse_layout_t dir,
                         int mb,
                         int nb,
                         int nnzb,
                         const alpha_matrix_descr_t descr_A,
                         const cuFloatComplex* bsr_val_A,
                         const int* bsr_row_ptr_A,
                         const int* bsr_col_ind_A,
                         int row_block_dim_A,
                         int col_block_dim_A,
                         const alpha_matrix_descr_t descr_C,
                         cuFloatComplex* bsr_val_C,
                         int* bsr_row_ptr_C,
                         int* bsr_col_ind_C,
                         int row_block_dim_C,
                         int col_block_dim_C,
                         void* temp_buffer);

alphasparseStatus_t
alphasparse_zgebsr2gebsr(alphasparseHandle_t handle,
                         alphasparse_layout_t dir,
                         int mb,
                         int nb,
                         int nnzb,
                         const alpha_matrix_descr_t descr_A,
                         const cuDoubleComplex* bsr_val_A,
                         const int* bsr_row_ptr_A,
                         const int* bsr_col_ind_A,
                         int row_block_dim_A,
                         int col_block_dim_A,
                         const alpha_matrix_descr_t descr_C,
                         cuDoubleComplex* bsr_val_C,
                         int* bsr_row_ptr_C,
                         int* bsr_col_ind_C,
                         int row_block_dim_C,
                         int col_block_dim_C,
                         void* temp_buffer);

alphasparseStatus_t
alphasparseAxpby(alphasparseHandle_t handle,
                 const void* alpha,
                 const alphasparseSpVecDescr_t x,
                 const void* beta,
                 alphasparseDnVecDescr_t y);

alphasparseStatus_t
alphasparseGather(alphasparseHandle_t handle,
                  const alphasparseDnVecDescr_t y,
                  alphasparseSpVecDescr_t x);

alphasparseStatus_t
alphasparseScatter(alphasparseHandle_t handle,
                   const alphasparseSpVecDescr_t x,
                   alphasparseDnVecDescr_t y);

alphasparseStatus_t
alphasparseRot(alphasparseHandle_t handle,
               const void* c,
               const void* s,
               alphasparseSpVecDescr_t x,
               alphasparseDnVecDescr_t y);

alphasparseStatus_t
alphasparseSpMV(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                const void* alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseDnVecDescr_t vecX,
                const void* beta,
                alphasparseDnVecDescr_t vecY,
                alphasparseDataType computeType,
                alphasparseSpMVAlg_t alg,
                void* externalBuffer);

alphasparseStatus_t
alphasparseSpMV_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnVecDescr_t vecX,
                           const void* beta,
                           alphasparseDnVecDescr_t vecY,
                           alphasparseDataType computeType,
                           alphasparseSpMVAlg_t alg,
                           size_t* bufferSize);

alphasparseStatus_t
alphasparseSpMV_analysis(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMVAlg_t alg);

alphasparseStatus_t
alphasparseSpMV_clear(alphasparseHandle_t handle, alphasparseSpMVAlg_t alg);

// alphasparseStatus_t
// alphasparseSpSV_solve(alphasparseHandle_t handle,
//                       alphasparseOperation_t opA,
//                       const void* alpha,
//                       alphasparseSpMatDescr_t matA,
//                       alphasparseDnVecDescr_t vecX,
//                       alphasparseDnVecDescr_t vecY,
//                       alphasparseDataType computeType,
//                       alphasparseSpSVAlg_t alg,
//                       alphasparseSpSVDescr_t spsvDescr);

alphasparseStatus_t
alphasparseSpSV_solve(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      const void* alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseDnVecDescr_t vecX,
                      alphasparseDnVecDescr_t vecY,
                      alphasparseDataType computeType,
                      alphasparseSpSVAlg_t alg,
                      alphasparseSpSVDescr_t spsvDescr,
                      void* externalBuffer);

alphasparseStatus_t
alphasparseSpSV_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnVecDescr_t vecX,
                           alphasparseDnVecDescr_t vecY,
                           alphasparseDataType computeType,
                           alphasparseSpSVAlg_t alg,
                           alphasparseSpSVDescr_t spsvDescr,
                           size_t* bufferSize);

alphasparseStatus_t
alphasparseSpMM(alphasparseHandle_t handle,
                alphasparseOperation_t opA,
                alphasparseOperation_t opB,
                const void* alpha,
                alphasparseSpMatDescr_t matA,
                alphasparseDnMatDescr_t matB,
                const void* beta,
                alphasparseDnMatDescr_t matC,
                alphasparseDataType computeType,
                alphasparseSpMMAlg_t alg,
                void* externalBuffer);

alphasparseStatus_t
alphasparseSpGEMM_compute(alphasparseHandle_t handle,
                          alphasparseOperation_t opA,
                          alphasparseOperation_t opB,
                          const void* alpha,
                          alphasparseSpMatDescr_t matA,
                          alphasparseSpMatDescr_t matB,
                          const void* beta,
                          alphasparseSpMatDescr_t matC,
                          alphasparseDataType computeType,
                          alphasparseSpGEMMAlg_t alg,
                          alphasparseSpGEMMDescr_t spgemmDescr,
                          size_t* bufferSize2,
                          void* externalBuffer2);

alphasparseStatus_t
alphasparseSpGEMMreuse_compute(alphasparseHandle_t handle,
                               alphasparseOperation_t opA,
                               alphasparseOperation_t opB,
                               const void* alpha,
                               alphasparseSpMatDescr_t matA,
                               alphasparseSpMatDescr_t matB,
                               const void* beta,
                               alphasparseSpMatDescr_t matC,
                               alphasparseDataType computeType,
                               alphasparseSpGEMMAlg_t alg,
                               alphasparseSpGEMMDescr_t spgemmDescr);

alphasparseStatus_t
alphasparseSpMM_preprocess(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           alphasparseOperation_t opB,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnMatDescr_t matB,
                           const void* beta,
                           alphasparseDnMatDescr_t matC,
                           alphasparseDataType computeType,
                           alphasparseSpMMAlg_t alg,
                           size_t* bufferSize);

alphasparseStatus_t
alphasparseSpMM_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           alphasparseOperation_t opB,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnMatDescr_t matB,
                           const void* beta,
                           alphasparseDnMatDescr_t matC,
                           alphasparseDataType computeType,
                           alphasparseSpMMAlg_t alg,
                           size_t* bufferSize);

alphasparseStatus_t
alphasparseSpSM_solve(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      alphasparseOperation_t opB,
                      const void* alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseDnMatDescr_t matB,
                      alphasparseDnMatDescr_t matC,
                      alphasparseDataType computeType,
                      alphasparseSpSMAlg_t alg,
                      alphasparseSpSMDescr_t spsmDescr);

alphasparseStatus_t
alphasparseSpSM_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           alphasparseOperation_t opB,
                           const void* alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnMatDescr_t matB,
                           alphasparseDnMatDescr_t matC,
                           alphasparseDataType computeType,
                           alphasparseSpSMAlg_t alg,
                           alphasparseSpSMDescr_t spsmDescr,
                           size_t* bufferSize);

alphasparseStatus_t
alphasparseSDDMM(alphasparseHandle_t handle,
                 alphasparseOperation_t opA,
                 alphasparseOperation_t opB,
                 const void* alpha,
                 alphasparseDnMatDescr_t matA,
                 alphasparseDnMatDescr_t matB,
                 const void* beta,
                 alphasparseSpMatDescr_t matC,
                 alphasparseDataType computeType,
                 alphasparseSDDMMAlg_t alg,
                 void* externalBuffer);

alphasparseStatus_t
alphasparseSDDMM_preprocess(alphasparseHandle_t handle,
                            alphasparseOperation_t opA,
                            alphasparseOperation_t opB,
                            const void* alpha,
                            alphasparseDnMatDescr_t matA,
                            alphasparseDnMatDescr_t matB,
                            const void* beta,
                            alphasparseSpMatDescr_t matC,
                            alphasparseDataType computeType,
                            alphasparseSDDMMAlg_t alg,
                            void* externalBuffer);

alphasparseStatus_t
alphasparseSDDMM_bufferSize(alphasparseHandle_t handle,
                            alphasparseOperation_t opA,
                            alphasparseOperation_t opB,
                            const void* alpha,
                            alphasparseDnMatDescr_t matA,
                            alphasparseDnMatDescr_t matB,
                            const void* beta,
                            alphasparseSpMatDescr_t matC,
                            alphasparseDataType computeType,
                            alphasparseSDDMMAlg_t alg,
                            size_t* bufferSize);

alphasparseStatus_t
alphasparseScsric02_bufferSize(alphasparseHandle_t handle,
                               int m,
                               int nnz,
                               const alphasparseMatDescr_t descrA,
                               float* csrValA,
                               const int* csrRowPtrA,
                               const int* csrColIndA,
                               alpha_csric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseScsric02_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alphasparseMatDescr_t descrA,
                             const float* csrValA,
                             const int* csrRowPtrA,
                             const int* csrColIndA,
                             alpha_csric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseScsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    float* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseDcsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    double* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseCcsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    void* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseZcsric02(alphasparseHandle_t handle,
                    int m,
                    int nnz,
                    const alphasparseMatDescr_t descrA,
                    void* csrValA_valM,
                    const int* csrRowPtrA,
                    const int* csrColIndA,
                    alpha_csric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseXcsric02_zeroPivot(alphasparseHandle_t handle,
                              alpha_csric02Info_t info,
                              int* position);

alphasparseStatus_t
alphasparseDcsric02_bufferSize(alphasparseHandle_t handle,
                               int m,
                               int nnz,
                               const alphasparseMatDescr_t descrA,
                               double* csrValA,
                               const int* csrRowPtrA,
                               const int* csrColIndA,
                               alpha_csric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDcsric02_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alphasparseMatDescr_t descrA,
                             const double* csrValA,
                             const int* csrRowPtrA,
                             const int* csrColIndA,
                             alpha_csric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseCcsric02_bufferSize(alphasparseHandle_t handle,
                               int m,
                               int nnz,
                               const alphasparseMatDescr_t descrA,
                               void* csrValA,
                               const int* csrRowPtrA,
                               const int* csrColIndA,
                               alpha_csric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCcsric02_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alphasparseMatDescr_t descrA,
                             const void* csrValA,
                             const int* csrRowPtrA,
                             const int* csrColIndA,
                             alpha_csric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseZcsric02_bufferSize(alphasparseHandle_t handle,
                               int m,
                               int nnz,
                               const alphasparseMatDescr_t descrA,
                               void* csrValA,
                               const int* csrRowPtrA,
                               const int* csrColIndA,
                               alpha_csric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZcsric02_analysis(alphasparseHandle_t handle,
                             int m,
                             int nnz,
                             const alphasparseMatDescr_t descrA,
                             const void* csrValA,
                             const int* csrRowPtrA,
                             const int* csrColIndA,
                             alpha_csric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseScsrilu02_bufferSize(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alphasparseMatDescr_t descrA,
                                float* csrValA,
                                const int* csrRowPtrA,
                                const int* csrColIndA,
                                alpha_csrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseScsrilu02_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alphasparseMatDescr_t descrA,
                              const float* csrValA,
                              const int* csrRowPtrA,
                              const int* csrColIndA,
                              alpha_csrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseScsrilu02(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     float* csrValA_valM,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     alpha_csrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseDcsrilu02(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     double* csrValA_valM,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     alpha_csrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseCcsrilu02(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     void* csrValA_valM,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     alpha_csrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseZcsrilu02(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     void* csrValA_valM,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     alpha_csrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseXcsrilu02_zeroPivot(alphasparseHandle_t handle,
                               alpha_csrilu02Info_t info,
                               int* position);

alphasparseStatus_t
alphasparseDcsrilu02_bufferSize(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alphasparseMatDescr_t descrA,
                                double* csrValA,
                                const int* csrRowPtrA,
                                const int* csrColIndA,
                                alpha_csrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDcsrilu02_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alphasparseMatDescr_t descrA,
                              const double* csrValA,
                              const int* csrRowPtrA,
                              const int* csrColIndA,
                              alpha_csrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseCcsrilu02_bufferSize(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alphasparseMatDescr_t descrA,
                                void* csrValA,
                                const int* csrRowPtrA,
                                const int* csrColIndA,
                                alpha_csrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCcsrilu02_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alphasparseMatDescr_t descrA,
                              const void* csrValA,
                              const int* csrRowPtrA,
                              const int* csrColIndA,
                              alpha_csrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseZcsrilu02_bufferSize(alphasparseHandle_t handle,
                                int m,
                                int nnz,
                                const alphasparseMatDescr_t descrA,
                                void* csrValA,
                                const int* csrRowPtrA,
                                const int* csrColIndA,
                                alpha_csrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZcsrilu02_analysis(alphasparseHandle_t handle,
                              int m,
                              int nnz,
                              const alphasparseMatDescr_t descrA,
                              const void* csrValA,
                              const int* csrRowPtrA,
                              const int* csrColIndA,
                              alpha_csrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseSbsrilu02_numericBoost(alphasparseHandle_t handle,
                                  alpha_bsrilu02Info_t info,
                                  int enable_boost,
                                  double* tol,
                                  float* boost_val);

alphasparseStatus_t
alphasparseDbsrilu02_numericBoost(alphasparseHandle_t handle,
                                  alpha_bsrilu02Info_t info,
                                  int enable_boost,
                                  double* tol,
                                  double* boost_val);

alphasparseStatus_t
alphasparseCbsrilu02_numericBoost(alphasparseHandle_t handle,
                                  alpha_bsrilu02Info_t info,
                                  int enable_boost,
                                  double* tol,
                                  void* boost_val);

alphasparseStatus_t
alphasparseZbsrilu02_numericBoost(alphasparseHandle_t handle,
                                  alpha_bsrilu02Info_t info,
                                  int enable_boost,
                                  double* tol,
                                  void* boost_val);

alphasparseStatus_t
alphasparseSbsrilu02_bufferSize(alphasparseHandle_t handle,
                                alphasparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const alphasparseMatDescr_t descrA,
                                float* bsrValA,
                                const int* bsrRowPtrA,
                                const int* bsrColIndA,
                                int blockDim,
                                alpha_bsrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSbsrilu02_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              const float* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseDbsrilu02_bufferSize(alphasparseHandle_t handle,
                                alphasparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const alphasparseMatDescr_t descrA,
                                double* bsrValA,
                                const int* bsrRowPtrA,
                                const int* bsrColIndA,
                                int blockDim,
                                alpha_bsrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDbsrilu02_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              const double* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseCbsrilu02_bufferSize(alphasparseHandle_t handle,
                                alphasparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const alphasparseMatDescr_t descrA,
                                void* bsrValA,
                                const int* bsrRowPtrA,
                                const int* bsrColIndA,
                                int blockDim,
                                alpha_bsrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCbsrilu02_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              const void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseZbsrilu02_bufferSize(alphasparseHandle_t handle,
                                alphasparseDirection_t dirA,
                                int mb,
                                int nnzb,
                                const alphasparseMatDescr_t descrA,
                                void* bsrValA,
                                const int* bsrRowPtrA,
                                const int* bsrColIndA,
                                int blockDim,
                                alpha_bsrilu02Info_t info,
                                int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZbsrilu02_analysis(alphasparseHandle_t handle,
                              alphasparseDirection_t dirA,
                              int mb,
                              int nnzb,
                              const alphasparseMatDescr_t descrA,
                              const void* bsrValA,
                              const int* bsrRowPtrA,
                              const int* bsrColIndA,
                              int blockDim,
                              alpha_bsrilu02Info_t info,
                              alphasparseSolvePolicy_t policy,
                              void* pBuffer);

alphasparseStatus_t
alphasparseSbsrilu02(alphasparseHandle_t handle,
                     alphasparseDirection_t dirA,
                     int mb,
                     int nnzb,
                     const alphasparseMatDescr_t descrA,
                     float* bsrValA,
                     const int* bsrRowPtrA,
                     const int* bsrColIndA,
                     int blockDim,
                     alpha_bsrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseDbsrilu02(alphasparseHandle_t handle,
                     alphasparseDirection_t dirA,
                     int mb,
                     int nnzb,
                     const alphasparseMatDescr_t descrA,
                     double* bsrValA,
                     const int* bsrRowPtrA,
                     const int* bsrColIndA,
                     int blockDim,
                     alpha_bsrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseCbsrilu02(alphasparseHandle_t handle,
                     alphasparseDirection_t dirA,
                     int mb,
                     int nnzb,
                     const alphasparseMatDescr_t descrA,
                     void* bsrValA,
                     const int* bsrRowPtrA,
                     const int* bsrColIndA,
                     int blockDim,
                     alpha_bsrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseZbsrilu02(alphasparseHandle_t handle,
                     alphasparseDirection_t dirA,
                     int mb,
                     int nnzb,
                     const alphasparseMatDescr_t descrA,
                     void* bsrValA,
                     const int* bsrRowPtrA,
                     const int* bsrColIndA,
                     int blockDim,
                     alpha_bsrilu02Info_t info,
                     alphasparseSolvePolicy_t policy,
                     void* pBuffer);

alphasparseStatus_t
alphasparseXbsrilu02_zeroPivot(alphasparseHandle_t handle,
                               alpha_bsrilu02Info_t info,
                               int* position);

alphasparseStatus_t
alphasparseSbsric02_bufferSize(alphasparseHandle_t handle,
                               alphasparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const alphasparseMatDescr_t descrA,
                               float* bsrValA,
                               const int* bsrRowPtrA,
                               const int* bsrColIndA,
                               int blockDim,
                               alpha_bsric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSbsric02_analysis(alphasparseHandle_t handle,
                             alphasparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const alphasparseMatDescr_t descrA,
                             const float* bsrValA,
                             const int* bsrRowPtrA,
                             const int* bsrColIndA,
                             int blockDim,
                             alpha_bsric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseDbsric02_bufferSize(alphasparseHandle_t handle,
                               alphasparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const alphasparseMatDescr_t descrA,
                               double* bsrValA,
                               const int* bsrRowPtrA,
                               const int* bsrColIndA,
                               int blockDim,
                               alpha_bsric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDbsric02_analysis(alphasparseHandle_t handle,
                             alphasparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const alphasparseMatDescr_t descrA,
                             const double* bsrValA,
                             const int* bsrRowPtrA,
                             const int* bsrColIndA,
                             int blockDim,
                             alpha_bsric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseCbsric02_bufferSize(alphasparseHandle_t handle,
                               alphasparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const alphasparseMatDescr_t descrA,
                               void* bsrValA,
                               const int* bsrRowPtrA,
                               const int* bsrColIndA,
                               int blockDim,
                               alpha_bsric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCbsric02_analysis(alphasparseHandle_t handle,
                             alphasparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const alphasparseMatDescr_t descrA,
                             const void* bsrValA,
                             const int* bsrRowPtrA,
                             const int* bsrColIndA,
                             int blockDim,
                             alpha_bsric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseZbsric02_bufferSize(alphasparseHandle_t handle,
                               alphasparseDirection_t dirA,
                               int mb,
                               int nnzb,
                               const alphasparseMatDescr_t descrA,
                               void* bsrValA,
                               const int* bsrRowPtrA,
                               const int* bsrColIndA,
                               int blockDim,
                               alpha_bsric02Info_t info,
                               int* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZbsric02_analysis(alphasparseHandle_t handle,
                             alphasparseDirection_t dirA,
                             int mb,
                             int nnzb,
                             const alphasparseMatDescr_t descrA,
                             const void* bsrValA,
                             const int* bsrRowPtrA,
                             const int* bsrColIndA,
                             int blockDim,
                             alpha_bsric02Info_t info,
                             alphasparseSolvePolicy_t policy,
                             void* pBuffer);

alphasparseStatus_t
alphasparseSbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    float* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseDbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    double* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseCbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    void* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseZbsric02(alphasparseHandle_t handle,
                    alphasparseDirection_t dirA,
                    int mb,
                    int nnzb,
                    const alphasparseMatDescr_t descrA,
                    void* bsrValA,
                    const int* bsrRowPtrA,
                    const int* bsrColIndA,
                    int blockDim,
                    alpha_bsric02Info_t info,
                    alphasparseSolvePolicy_t policy,
                    void* pBuffer);

alphasparseStatus_t
alphasparseXbsric02_zeroPivot(alphasparseHandle_t handle,
                              alpha_bsric02Info_t info,
                              int* position);

alphasparseStatus_t
alphasparseSgtsv2_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const float* dl,
                                const float* d,
                                const float* du,
                                float* B,
                                int ldb,
                                size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseDgtsv2_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const double* dl,
                                const double* d,
                                const double* du,
                                double* B,
                                int ldb,
                                size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseCgtsv2_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const cuFloatComplex* dl,
                                const cuFloatComplex* d,
                                const cuFloatComplex* du,
                                cuFloatComplex* B,
                                int ldb,
                                size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseZgtsv2_bufferSizeExt(alphasparseHandle_t handle,
                                int m,
                                int n,
                                const cuDoubleComplex* dl,
                                const cuDoubleComplex* d,
                                const cuDoubleComplex* du,
                                cuDoubleComplex* B,
                                int ldb,
                                size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseSgtsv2(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const float* dl,
                  const float* d,
                  const float* du,
                  float* B,
                  int ldb,
                  void* pBuffer);

alphasparseStatus_t
alphasparseDgtsv2(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const double* dl,
                  const double* d,
                  const double* du,
                  double* B,
                  int ldb,
                  void* pBuffer);

alphasparseStatus_t
alphasparseCgtsv2(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const cuFloatComplex* dl,
                  const cuFloatComplex* d,
                  const cuFloatComplex* du,
                  cuFloatComplex* B,
                  int ldb,
                  void* pBuffer);

alphasparseStatus_t
alphasparseZgtsv2(alphasparseHandle_t handle,
                  int m,
                  int n,
                  const cuDoubleComplex* dl,
                  const cuDoubleComplex* d,
                  const cuDoubleComplex* du,
                  cuDoubleComplex* B,
                  int ldb,
                  void* pBuffer);

alphasparseStatus_t
alphasparseSgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                        int m,
                                        int n,
                                        const float* dl,
                                        const float* d,
                                        const float* du,
                                        float* B,
                                        int ldb,
                                        size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseDgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                        int m,
                                        int n,
                                        const double* dl,
                                        const double* d,
                                        const double* du,
                                        double* B,
                                        int ldb,
                                        size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseCgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                        int m,
                                        int n,
                                        const cuFloatComplex* dl,
                                        const cuFloatComplex* d,
                                        const cuFloatComplex* du,
                                        cuFloatComplex* B,
                                        int ldb,
                                        size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseZgtsv2_nopivot_bufferSizeExt(alphasparseHandle_t handle,
                                        int m,
                                        int n,
                                        const cuDoubleComplex* dl,
                                        const cuDoubleComplex* d,
                                        const cuDoubleComplex* du,
                                        cuDoubleComplex* B,
                                        int ldb,
                                        size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseSgtsv2_nopivot(alphasparseHandle_t handle,
                          int m,
                          int n,
                          const float* dl,
                          const float* d,
                          const float* du,
                          float* B,
                          int ldb,
                          void* pBuffer);

alphasparseStatus_t
alphasparseDgtsv2_nopivot(alphasparseHandle_t handle,
                          int m,
                          int n,
                          const double* dl,
                          const double* d,
                          const double* du,
                          double* B,
                          int ldb,
                          void* pBuffer);

alphasparseStatus_t
alphasparseCgtsv2_nopivot(alphasparseHandle_t handle,
                          int m,
                          int n,
                          const cuFloatComplex* dl,
                          const cuFloatComplex* d,
                          const cuFloatComplex* du,
                          cuFloatComplex* B,
                          int ldb,
                          void* pBuffer);

alphasparseStatus_t
alphasparseZgtsv2_nopivot(alphasparseHandle_t handle,
                          int m,
                          int n,
                          const cuDoubleComplex* dl,
                          const cuDoubleComplex* d,
                          const cuDoubleComplex* du,
                          cuDoubleComplex* B,
                          int ldb,
                          void* pBuffer);

alphasparseStatus_t
alphasparseSgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const float* dl,
                                            const float* d,
                                            const float* du,
                                            const float* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseDgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const double* dl,
                                            const double* d,
                                            const double* du,
                                            const double* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseCgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const cuFloatComplex* dl,
                                            const cuFloatComplex* d,
                                            const cuFloatComplex* du,
                                            const cuFloatComplex* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseZgtsv2StridedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                            int m,
                                            const cuDoubleComplex* dl,
                                            const cuDoubleComplex* d,
                                            const cuDoubleComplex* du,
                                            const cuDoubleComplex* x,
                                            int batchCount,
                                            int batchStride,
                                            size_t* bufferSizeInBytes);

alphasparseStatus_t
alphasparseSgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const float* dl,
                              const float* d,
                              const float* du,
                              float* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer);

alphasparseStatus_t
alphasparseDgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const double* dl,
                              const double* d,
                              const double* du,
                              double* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer);

alphasparseStatus_t
alphasparseCgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const cuFloatComplex* dl,
                              const cuFloatComplex* d,
                              const cuFloatComplex* du,
                              cuFloatComplex* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer);

alphasparseStatus_t
alphasparseZgtsv2StridedBatch(alphasparseHandle_t handle,
                              int m,
                              const cuDoubleComplex* dl,
                              const cuDoubleComplex* d,
                              const cuDoubleComplex* du,
                              cuDoubleComplex* x,
                              int batchCount,
                              int batchStride,
                              void* pBuffer);

alphasparseStatus_t
alphasparseSgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const float* dl,
                                               const float* d,
                                               const float* du,
                                               const float* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const double* dl,
                                               const double* d,
                                               const double* du,
                                               const double* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const cuFloatComplex* dl,
                                               const cuFloatComplex* d,
                                               const cuFloatComplex* du,
                                               const cuFloatComplex* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZgtsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const cuDoubleComplex* dl,
                                               const cuDoubleComplex* d,
                                               const cuDoubleComplex* du,
                                               const cuDoubleComplex* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 float* dl,
                                 float* d,
                                 float* du,
                                 float* x,
                                 int batchCount,
                                 void* pBuffer);

alphasparseStatus_t
alphasparseDgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 double* dl,
                                 double* d,
                                 double* du,
                                 double* x,
                                 int batchCount,
                                 void* pBuffer);

alphasparseStatus_t
alphasparseCgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 cuFloatComplex* dl,
                                 cuFloatComplex* d,
                                 cuFloatComplex* du,
                                 cuFloatComplex* x,
                                 int batchCount,
                                 void* pBuffer);

alphasparseStatus_t
alphasparseZgtsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 cuDoubleComplex* dl,
                                 cuDoubleComplex* d,
                                 cuDoubleComplex* du,
                                 cuDoubleComplex* x,
                                 int batchCount,
                                 void* pBuffer);

alphasparseStatus_t
alphasparse_sparse_to_dense(alphasparseHandle_t handle,
                            const alphasparse_spmat_descr_t mat_A,
                            alphasparse_dnmat_descr_t mat_B,
                            alphasparse_sparse_to_dense_alg_t alg,
                            size_t* buffer_size,
                            void* temp_buffer);

alphasparseStatus_t
alphasparseScsrcolor(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     const float* csrValA,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     const float* fractionToColor,
                     int* ncolors,
                     int* coloring,
                     int* reordering,
                     alphasparseColorInfo_t info);

alphasparseStatus_t
alphasparseDcsrcolor(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     const double* csrValA,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     const double* fractionToColor,
                     int* ncolors,
                     int* coloring,
                     int* reordering,
                     alphasparseColorInfo_t info);

alphasparseStatus_t
alphasparseCcsrcolor(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     const void* csrValA,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     const float* fractionToColor,
                     int* ncolors,
                     int* coloring,
                     int* reordering,
                     alphasparseColorInfo_t info);

alphasparseStatus_t
alphasparseZcsrcolor(alphasparseHandle_t handle,
                     int m,
                     int nnz,
                     const alphasparseMatDescr_t descrA,
                     const void* csrValA,
                     const int* csrRowPtrA,
                     const int* csrColIndA,
                     const double* fractionToColor,
                     int* ncolors,
                     int* coloring,
                     int* reordering,
                     alphasparseColorInfo_t info);

alphasparseStatus_t
alphasparseSgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const float* ds,
                                               const float* dl,
                                               const float* d,
                                               const float* du,
                                               const float* dw,
                                               const float* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseDgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const double* ds,
                                               const double* dl,
                                               const double* d,
                                               const double* du,
                                               const double* dw,
                                               const double* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseCgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const void* ds,
                                               const void* dl,
                                               const void* d,
                                               const void* du,
                                               const void* dw,
                                               const void* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseZgpsvInterleavedBatch_bufferSizeExt(alphasparseHandle_t handle,
                                               int algo,
                                               int m,
                                               const void* ds,
                                               const void* dl,
                                               const void* d,
                                               const void* du,
                                               const void* dw,
                                               const void* x,
                                               int batchCount,
                                               size_t* pBufferSizeInBytes);

alphasparseStatus_t
alphasparseSgpsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 float* ds,
                                 float* dl,
                                 float* d,
                                 float* du,
                                 float* dw,
                                 float* x,
                                 int batchCount,
                                 void* pBuffer);

alphasparseStatus_t
alphasparseDgpsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 double* ds,
                                 double* dl,
                                 double* d,
                                 double* du,
                                 double* dw,
                                 double* x,
                                 int batchCount,
                                 void* pBuffer);

alphasparseStatus_t
alphasparseCgpsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 void* ds,
                                 void* dl,
                                 void* d,
                                 void* du,
                                 void* dw,
                                 void* x,
                                 int batchCount,
                                 void* pBuffer);
alphasparseStatus_t
alphasparseZgpsvInterleavedBatch(alphasparseHandle_t handle,
                                 int algo,
                                 int m,
                                 void* ds,
                                 void* dl,
                                 void* d,
                                 void* du,
                                 void* dw,
                                 void* x,
                                 int batchCount,
                                 void* pBuffer);
alphasparseStatus_t
alphasparse_dense_to_sparse(alphasparseHandle_t handle,
                            const alphasparse_dnmat_descr_t mat_A,
                            alphasparse_spmat_descr_t mat_B,
                            alphasparse_dense_to_sparse_alg_t alg,
                            size_t* buffer_size,
                            void* temp_buffer);

alphasparseStatus_t
alphasparseSpvv(alphasparseHandle_t handle,
                alphasparseOperation_t trans,
                const alphasparseSpVecDescr_t x,
                const alphasparseDnVecDescr_t y,
                void* result,
                alphasparseDataType compute_type,
                size_t* buffer_size,
                void* temp_buffer);

alphasparseStatus_t
alphasparse_spgemm(alphasparseHandle_t handle,
                   alphasparseOperation_t trans_A,
                   alphasparseOperation_t trans_B,
                   const void* alpha,
                   const alphasparse_spmat_descr_t A,
                   const alphasparse_spmat_descr_t B,
                   const void* beta,
                   const alphasparse_spmat_descr_t D,
                   alphasparse_spmat_descr_t C,
                   alphasparseDataType compute_type,
                   alphasparse_spgemm_alg_t alg,
                   alphasparse_spgemm_stage_t stage,
                   size_t* buffer_size,
                   void* temp_buffer);

/**
 * convert from host point to device point
 *
 */
alphasparseStatus_t
host2device_h_csr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_s_csr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_csr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_csr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_csr(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_csr5(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_csr5(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_csr5(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_csr5(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_coo(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_coo(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_coo(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_coo(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_ell(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_ell(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_ell(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_ell(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_bsr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_bsr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_bsr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_bsr(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_gebsr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_gebsr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_gebsr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_gebsr(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_hyb(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_hyb(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_hyb(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_hyb(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_cooaos(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_cooaos(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_cooaos(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_cooaos(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_sell_csigma(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_sell_csigma(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_sell_csigma(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_sell_csigma(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_ellr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_ellr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_ellr(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_ellr(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_csc(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_csc(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_csc(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_csc(alphasparse_matrix_t A);

alphasparseStatus_t
host2device_s_dia(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_d_dia(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_c_dia(alphasparse_matrix_t A);
alphasparseStatus_t
host2device_z_dia(alphasparse_matrix_t A);

/**
 * general format for spmv
 *
 */
#ifdef __CUDA__
alphasparseStatus_t
alphasparse_h_mv(alphasparseHandle_t handle,
                 const alphasparseOperation_t operation,
                 const half* alpha,
                 const alphasparse_matrix_t A,
                 const alpha_matrix_descr_t descr,
                 const half* x,
                 const half* beta,
                 half* y);
#endif
alphasparseStatus_t
alphasparse_s_mv(alphasparseHandle_t handle,
                 const alphasparseOperation_t operation,
                 const float* alpha,
                 const alphasparse_matrix_t A,
                 const alpha_matrix_descr_t descr,
                 const float* x,
                 const float* beta,
                 float* y);

alphasparseStatus_t
alphasparse_d_mv(alphasparseHandle_t handle,
                 const alphasparseOperation_t operation,
                 const double* alpha,
                 const alphasparse_matrix_t A,
                 const alpha_matrix_descr_t descr,
                 const double* x,
                 const double* beta,
                 double* y);

alphasparseStatus_t
alphasparse_c_mv(alphasparseHandle_t handle,
                 const alphasparseOperation_t operation,
                 const cuFloatComplex* alpha,
                 const alphasparse_matrix_t A,
                 const alpha_matrix_descr_t descr,
                 const cuFloatComplex* x,
                 const cuFloatComplex* beta,
                 cuFloatComplex* y);

alphasparseStatus_t
alphasparse_z_mv(alphasparseHandle_t handle,
                 const alphasparseOperation_t operation,
                 const cuDoubleComplex* alpha,
                 const alphasparse_matrix_t A,
                 const alpha_matrix_descr_t descr,
                 const cuDoubleComplex* x,
                 const cuDoubleComplex* beta,
                 cuDoubleComplex* y);

alphasparseStatus_t
alphasparseAxpby(alphasparseHandle_t handle,
                 const void* alpha,
                 const alphasparseSpVecDescr_t x,
                 const void* beta,
                 alphasparseDnVecDescr_t y);

alphasparseStatus_t
alphasparseGather(alphasparseHandle_t handle,
                  const alphasparseDnVecDescr_t y,
                  alphasparseSpVecDescr_t x);

alphasparseStatus_t
alphasparseScatter(alphasparseHandle_t handle,
                   const alphasparseSpVecDescr_t x,
                   alphasparseDnVecDescr_t y);

alphasparseStatus_t
alphasparseRot(alphasparseHandle_t handle,
               const void* c,
               const void* s,
               alphasparseSpVecDescr_t x,
               alphasparseDnVecDescr_t y);

alphasparseStatus_t
alphasparseSpvv(alphasparseHandle_t handle,
                alphasparseOperation_t trans,
                const alphasparseSpVecDescr_t x,
                const alphasparseDnVecDescr_t y,
                void* result,
                alphasparseDataType compute_type,
                void* temp_buffer);

alphasparseStatus_t
alphasparseSpvv_buffersize(alphasparseHandle_t handle,
                           alphasparseOperation_t trans,
                           const alphasparseSpVecDescr_t x,
                           const alphasparseDnVecDescr_t y,
                           void* result,
                           alphasparseDataType compute_type,
                           size_t* buffer_size);

alphasparseStatus_t
alphasparseSpSV_analysis(alphasparseHandle_t handle,
                         alphasparseOperation_t opA,
                         const void* alpha,
                         alphasparseSpMatDescr_t matA,
                         alphasparseDnVecDescr_t vecX,
                         alphasparseDnVecDescr_t vecY,
                         alphasparseDataType computeType,
                         alphasparseSpSVAlg_t alg,
                         alphasparseSpSVDescr_t spsvDescr,
                         void* externalBuffer);
#endif

//old version
alphasparseStatus_t alphasparse_transpose(const alphasparse_matrix_t source, alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_csc(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_bsr(const alphasparse_matrix_t source, /* convert original matrix to BSR representation */
                                           const ALPHA_INT block_size,
                                           const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                           const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_sky(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           const alphasparse_fill_mode_t fill,
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_dia(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_ell(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_hyb(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_csr5(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_cooaos(const alphasparse_matrix_t source,
                                           const alphasparseOperation_t operation,
                                           alphasparse_matrix_t *dest);
alphasparseStatus_t alphasparse_convert_sell_csigma(
                                                const alphasparse_matrix_t source, /* convert original matrix to SELL_C_Sgima representation */
                                                const bool SHORT_BINNING, const ALPHA_INT C, 
                                                const ALPHA_INT SIGMA,
                                                const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                                alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_ellr(const alphasparse_matrix_t source,
                                              const alphasparseOperation_t operation,
                                              alphasparse_matrix_t *dest);


alphasparseStatus_t alphasparse_convert_hints_bsr(const alphasparse_matrix_t source, /* convert original matrix to BSR representation */
                                                 const ALPHA_INT block_size,
                                                 const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                                 const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                                 alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_hints_dia(const alphasparse_matrix_t source,
                                                 const alphasparseOperation_t operation,
                                                 alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_hints_ell(const alphasparse_matrix_t source,
                                                 const alphasparseOperation_t operation,
                                                 alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_gebsr(const alphasparse_matrix_t source, /* convert original matrix to GEBSR representation */
                                             const ALPHA_INT block_row_dim,
                                             const ALPHA_INT block_col_dim,
                                             const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                             const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                             alphasparse_matrix_t *dest);

/**
 * --------------------------------------------------------------------------------------
 */

/*****************************************************************************************/
/*************************************** Creation routines *******************************/
/*****************************************************************************************/

/*
    Matrix handle is used for storing information about the matrix and matrix values

    Create matrix from one of the existing sparse formats by creating the handle with matrix info and copy matrix values if requested.
    Collect high-level info about the matrix. Need to use this interface for the case with several calls in program for performance reasons,
    where optimizations are not required.

    coordinate format,
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
alphasparseStatus_t alphasparse_s_create_coo(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            float *values);

alphasparseStatus_t alphasparse_d_create_coo(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            double *values);

alphasparseStatus_t alphasparse_c_create_coo(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex8 *values);

alphasparseStatus_t alphasparse_z_create_coo(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT nnz,
                                            ALPHA_INT *row_indx,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex16 *values);

/*
    compressed sparse row format (4-arrays version),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.


*/
alphasparseStatus_t alphasparse_s_create_csr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            float *values);

alphasparseStatus_t alphasparse_d_create_csr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            double *values);

alphasparseStatus_t alphasparse_c_create_csr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex8 *values);

alphasparseStatus_t alphasparse_z_create_csr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex16 *values);

/*
    compressed sparse column format (4-arrays version),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
alphasparseStatus_t alphasparse_s_create_csc(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *cols_start,
                                            ALPHA_INT *cols_end,
                                            ALPHA_INT *row_indx,
                                            float *values);

alphasparseStatus_t alphasparse_d_create_csc(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *cols_start,
                                            ALPHA_INT *cols_end,
                                            ALPHA_INT *row_indx,
                                            double *values);

alphasparseStatus_t alphasparse_c_create_csc(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *cols_start,
                                            ALPHA_INT *cols_end,
                                            ALPHA_INT *row_indx,
                                            ALPHA_Complex8 *values);

alphasparseStatus_t alphasparse_z_create_csc(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            ALPHA_INT *cols_start,
                                            ALPHA_INT *cols_end,
                                            ALPHA_INT *row_indx,
                                            ALPHA_Complex16 *values);

/*
    compressed block sparse row format (4-arrays version, square blocks),
    ALPHA_SPARSE_MATRIX_TYPE_GENERAL by default, pointers to input arrays are stored in the handle

    *** User data is not marked const since the alphasparse_order() or alphasparse_?_set_values()
    functionality could change user data.  However, this is only done by a user call. 
    Internally const-ness of user data is maintained other than through explicit
    use of these interfaces.

*/
alphasparseStatus_t alphasparse_s_create_bsr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT block_size,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            float *values);

alphasparseStatus_t alphasparse_d_create_bsr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT block_size,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            double *values);

alphasparseStatus_t alphasparse_c_create_bsr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT block_size,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex8 *values);

alphasparseStatus_t alphasparse_z_create_bsr(alphasparse_matrix_t *A,
                                            const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
                                            const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
                                            const ALPHA_INT rows,
                                            const ALPHA_INT cols,
                                            const ALPHA_INT block_size,
                                            ALPHA_INT *rows_start,
                                            ALPHA_INT *rows_end,
                                            ALPHA_INT *col_indx,
                                            ALPHA_Complex16 *values);

/*
    Create copy of the existing handle; matrix properties could be changed.
    For example it could be used for extracting triangular or diagonal parts from existing matrix.
*/
alphasparseStatus_t alphasparse_copy(const alphasparse_matrix_t source,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    alphasparse_matrix_t *dest);

/*
    destroy matrix handle; if sparse matrix was stored inside the handle it also deallocates the matrix
    It is user's responsibility not to delete the handle with the matrix, if this matrix is shared with other handles
*/
alphasparseStatus_t alphasparse_destroy(alphasparse_matrix_t A);
/*
    return extended error information from last operation;
    e.g. info about wrong input parameter, memory sizes that couldn't be allocated
*/
alphasparseStatus_t alphasparse_get_error_info(alphasparse_matrix_t A, ALPHA_INT *info); /* unsupported currently */

/*****************************************************************************************/
/************************ Converters of internal representation  *************************/
/*****************************************************************************************/

/* converters from current format to another */
alphasparseStatus_t alphasparse_convert_csr(const alphasparse_matrix_t source,       /* convert original matrix to CSR representation */
                                           const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_convert_coo(const alphasparse_matrix_t source,       /* convert original matrix to CSR representation */
                                           const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                           alphasparse_matrix_t *dest);

alphasparseStatus_t alphasparse_s_export_coo(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **row_indx,
                                            ALPHA_INT **col_indx,
                                            float **values,
                                            ALPHA_INT *nnz);

alphasparseStatus_t alphasparse_d_export_coo(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **row_indx,
                                            ALPHA_INT **col_indx,
                                            double **values,
                                            ALPHA_INT *nnz);

alphasparseStatus_t alphasparse_c_export_coo(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **row_indx,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex8 **values,
                                            ALPHA_INT *nnz);

alphasparseStatus_t alphasparse_z_export_coo(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **row_indx,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex16 **values,
                                            ALPHA_INT *nnz);

alphasparseStatus_t alphasparse_s_export_bsr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *block_size,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            float **values);

alphasparseStatus_t alphasparse_d_export_bsr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *block_size,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            double **values);

alphasparseStatus_t alphasparse_c_export_bsr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *block_size,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex8 **values);

alphasparseStatus_t alphasparse_z_export_bsr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *block_size,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex16 **values);

alphasparseStatus_t alphasparse_s_export_csr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            float **values);

alphasparseStatus_t alphasparse_d_export_csr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            double **values);

alphasparseStatus_t alphasparse_c_export_csr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex8 **values);

alphasparseStatus_t alphasparse_z_export_csr(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **rows_start,
                                            ALPHA_INT **rows_end,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex16 **values);

alphasparseStatus_t alphasparse_s_export_csc(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **cols_start,
                                            ALPHA_INT **cols_end,
                                            ALPHA_INT **row_indx,
                                            float **values);

alphasparseStatus_t alphasparse_d_export_csc(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **cols_start,
                                            ALPHA_INT **cols_end,
                                            ALPHA_INT **row_indx,
                                            double **values);

alphasparseStatus_t alphasparse_c_export_csc(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **cols_start,
                                            ALPHA_INT **cols_end,
                                            ALPHA_INT **row_indx,
                                            ALPHA_Complex8 **values);

alphasparseStatus_t alphasparse_z_export_csc(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT **cols_start,
                                            ALPHA_INT **cols_end,
                                            ALPHA_INT **row_indx,
                                            ALPHA_Complex16 **values);

alphasparseStatus_t alphasparse_s_export_ell(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *width,
                                            ALPHA_INT **col_indx,
                                            float **values);

alphasparseStatus_t alphasparse_d_export_ell(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *width,
                                            ALPHA_INT **col_indx,
                                            double **values);

alphasparseStatus_t alphasparse_c_export_ell(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *width,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex8 **values);

alphasparseStatus_t alphasparse_z_export_ell(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *width,
                                            ALPHA_INT **col_indx,
                                            ALPHA_Complex16 **values);

alphasparseStatus_t alphasparse_s_export_gebsr(const alphasparse_matrix_t source,
                                              alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                              alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                              ALPHA_INT *rows,
                                              ALPHA_INT *cols,
                                              ALPHA_INT *block_row_dim,
                                              ALPHA_INT *block_col_dim,
                                              ALPHA_INT **rows_start,
                                              ALPHA_INT **rows_end,
                                              ALPHA_INT **col_indx,
                                              float **values);

alphasparseStatus_t alphasparse_d_export_gebsr(const alphasparse_matrix_t source,
                                              alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                              alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                              ALPHA_INT *rows,
                                              ALPHA_INT *cols,
                                              ALPHA_INT *block_row_dim,
                                              ALPHA_INT *block_col_dim,
                                              ALPHA_INT **rows_start,
                                              ALPHA_INT **rows_end,
                                              ALPHA_INT **col_indx,
                                              double **values);

alphasparseStatus_t alphasparse_c_export_gebsr(const alphasparse_matrix_t source,
                                              alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                              alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                              ALPHA_INT *rows,
                                              ALPHA_INT *cols,
                                              ALPHA_INT *block_row_dim,
                                              ALPHA_INT *block_col_dim,
                                              ALPHA_INT **rows_start,
                                              ALPHA_INT **rows_end,
                                              ALPHA_INT **col_indx,
                                              ALPHA_Complex8 **values);

alphasparseStatus_t alphasparse_z_export_gebsr(const alphasparse_matrix_t source,
                                              alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                              alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                                              ALPHA_INT *rows,
                                              ALPHA_INT *cols,
                                              ALPHA_INT *block_row_dim,
                                              ALPHA_INT *block_col_dim,
                                              ALPHA_INT **rows_start,
                                              ALPHA_INT **rows_end,
                                              ALPHA_INT **col_indx,
                                              ALPHA_Complex16 **values);

alphasparseStatus_t alphasparse_s_export_hyb(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *nnz,
                                            ALPHA_INT *ell_width,
                                            float **ell_val,
                                            ALPHA_INT **ell_col_ind,
                                            float **coo_val,
                                            ALPHA_INT **coo_row_val,
                                            ALPHA_INT **coo_col_val);

alphasparseStatus_t alphasparse_d_export_hyb(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *nnz,
                                            ALPHA_INT *ell_width,
                                            double **ell_val,
                                            ALPHA_INT **ell_col_ind,
                                            double **coo_val,
                                            ALPHA_INT **coo_row_val,
                                            ALPHA_INT **coo_col_val);

alphasparseStatus_t alphasparse_c_export_hyb(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *nnz,
                                            ALPHA_INT *ell_width,
                                            ALPHA_Complex8 **ell_val,
                                            ALPHA_INT **ell_col_ind,
                                            ALPHA_Complex8 **coo_val,
                                            ALPHA_INT **coo_row_val,
                                            ALPHA_INT **coo_col_val);

alphasparseStatus_t alphasparse_z_export_hyb(const alphasparse_matrix_t source,
                                            alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                                            ALPHA_INT *rows,
                                            ALPHA_INT *cols,
                                            ALPHA_INT *nnz,
                                            ALPHA_INT *ell_width,
                                            ALPHA_Complex16 **ell_val,
                                            ALPHA_INT **ell_col_ind,
                                            ALPHA_Complex16 **coo_val,
                                            ALPHA_INT **coo_row_val,
                                            ALPHA_INT **coo_col_val);

/*****************************************************************************************/
/************************** Step-by-step modification routines ***************************/
/*****************************************************************************************/

/* update existing value in the matrix ( for internal storage only, should not work with user-allocated matrices) */
alphasparseStatus_t alphasparse_s_set_value(alphasparse_matrix_t A,
                                           const ALPHA_INT row,
                                           const ALPHA_INT col,
                                           const float value);

alphasparseStatus_t alphasparse_d_set_value(alphasparse_matrix_t A,
                                           const ALPHA_INT row,
                                           const ALPHA_INT col,
                                           const double value);

alphasparseStatus_t alphasparse_c_set_value(alphasparse_matrix_t A,
                                           const ALPHA_INT row,
                                           const ALPHA_INT col,
                                           const ALPHA_Complex8 value);

alphasparseStatus_t alphasparse_z_set_value(alphasparse_matrix_t A,
                                           const ALPHA_INT row,
                                           const ALPHA_INT col,
                                           const ALPHA_Complex16 value);

/* update existing values in the matrix for internal storage only 
       can be used to either update all or selected values */
alphasparseStatus_t alphasparse_s_update_values(alphasparse_matrix_t A,
                                               const ALPHA_INT nvalues,
                                               const ALPHA_INT *indx,
                                               const ALPHA_INT *indy,
                                               float *values);

alphasparseStatus_t alphasparse_d_update_values(alphasparse_matrix_t A,
                                               const ALPHA_INT nvalues,
                                               const ALPHA_INT *indx,
                                               const ALPHA_INT *indy,
                                               double *values);

alphasparseStatus_t alphasparse_c_update_values(alphasparse_matrix_t A,
                                               const ALPHA_INT nvalues,
                                               const ALPHA_INT *indx,
                                               const ALPHA_INT *indy,
                                               ALPHA_Complex8 *values);

alphasparseStatus_t alphasparse_z_update_values(alphasparse_matrix_t A,
                                               const ALPHA_INT nvalues,
                                               const ALPHA_INT *indx,
                                               const ALPHA_INT *indy,
                                               ALPHA_Complex16 *values);

/*****************************************************************************************/
/****************************** Verbose mode routine *************************************/
/*****************************************************************************************/

/* allow to switch on/off verbose mode */
alphasparseStatus_t alphasparse_set_verbose_mode(alpha_verbose_mode_t verbose); /* unsupported currently */

/*****************************************************************************************/
/****************************** Optimization routines ************************************/
/*****************************************************************************************/

/* Describe expected operations with amount of iterations */
alphasparseStatus_t alphasparse_set_mv_hint(const alphasparse_matrix_t A,
                                           const alphasparseOperation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                           const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                           const ALPHA_INT expected_calls);

alphasparseStatus_t alphasparse_set_dotmv_hint(const alphasparse_matrix_t A,
                                              const alphasparseOperation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                              const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                              const ALPHA_INT expectedCalls);

alphasparseStatus_t alphasparse_set_mmd_hint(const alphasparse_matrix_t A,
                                            const alphasparseOperation_t operation,
                                            const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                            const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                            const ALPHA_INT dense_matrix_size,     /* amount of columns in dense matrix */
                                            const ALPHA_INT expected_calls);

alphasparseStatus_t alphasparse_set_sv_hint(const alphasparse_matrix_t A,
                                           const alphasparseOperation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                           const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                           const ALPHA_INT expected_calls);

alphasparseStatus_t alphasparse_set_sm_hint(const alphasparse_matrix_t A,
                                           const alphasparseOperation_t operation,
                                           const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                           const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                           const ALPHA_INT dense_matrix_size,     /* amount of columns in dense matrix */
                                           const ALPHA_INT expected_calls);

alphasparseStatus_t alphasparse_set_mm_hint(const alphasparse_matrix_t A,
                                           const alphasparseOperation_t transA,
                                           const struct alpha_matrix_descr descrA, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                           const alphasparse_matrix_t B,
                                           const alphasparseOperation_t transB,
                                           const struct alpha_matrix_descr descrB,
                                           const ALPHA_INT expected_calls);

alphasparseStatus_t alphasparse_set_symgs_hint(const alphasparse_matrix_t A,
                                              const alphasparseOperation_t operation, /* ALPHA_SPARSE_OPERATION_NON_TRANSPOSE is default value for infinite amount of calls */
                                              const struct alpha_matrix_descr descr,    /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                              const ALPHA_INT expected_calls);

alphasparseStatus_t alphasparse_set_lu_smoother_hint(const alphasparse_matrix_t A,
                                                    const alphasparseOperation_t operation,
                                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                                    const ALPHA_INT expectedCalls);

/* Describe memory usage model */
alphasparseStatus_t alphasparse_set_memory_hint(const alphasparse_matrix_t A,
                                               const alphasparse_memory_usage_t policy); /* ALPHA_SPARSE_MEMORY_AGGRESSIVE is default value */

/*
    Optimize matrix described by the handle. It uses hints (optimization and memory) that should be set up before this call.
    If hints were not explicitly defined, default vales are:
    ALPHA_SPARSE_OPERATION_NON_TRANSPOSE for matrix-vector multiply with infinite number of expected iterations.
*/
alphasparseStatus_t alphasparse_optimize(alphasparse_matrix_t A);

/*****************************************************************************************/
/****************************** Computational routines ***********************************/
/*****************************************************************************************/

alphasparseStatus_t alphasparse_order(const alphasparse_matrix_t A);

/*
    Perform computations based on created matrix handle

    Level 2
*/
/*   Computes y = alpha * A * x + beta * y   */
alphasparseStatus_t alphasparse_s_mv(const alphasparseOperation_t operation,
                                    const float alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const float *x,
                                    const float beta,
                                    float *y);

alphasparseStatus_t alphasparse_d_mv(const alphasparseOperation_t operation,
                                    const double alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const double *x,
                                    const double beta,
                                    double *y);

alphasparseStatus_t alphasparse_c_mv(const alphasparseOperation_t operation,
                                    const ALPHA_Complex8 alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const ALPHA_Complex8 *x,
                                    const ALPHA_Complex8 beta,
                                    ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_mv(const alphasparseOperation_t operation,
                                    const ALPHA_Complex16 alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const ALPHA_Complex16 *x,
                                    const ALPHA_Complex16 beta,
                                    ALPHA_Complex16 *y);

/*    Computes y = alpha * A * x + beta * y  and d = <x, y> , the l2 inner product */
alphasparseStatus_t alphasparse_s_dotmv(const alphasparseOperation_t transA,
                                       const float alpha,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                       const float *x,
                                       const float beta,
                                       float *y,
                                       float *d);

alphasparseStatus_t alphasparse_d_dotmv(const alphasparseOperation_t transA,
                                       const double alpha,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                       const double *x,
                                       const double beta,
                                       double *y,
                                       double *d);

alphasparseStatus_t alphasparse_c_dotmv(const alphasparseOperation_t transA,
                                       const ALPHA_Complex8 alpha,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                       const ALPHA_Complex8 *x,
                                       const ALPHA_Complex8 beta,
                                       ALPHA_Complex8 *y,
                                       ALPHA_Complex8 *d);

alphasparseStatus_t alphasparse_z_dotmv(const alphasparseOperation_t transA,
                                       const ALPHA_Complex16 alpha,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                       const ALPHA_Complex16 *x,
                                       const ALPHA_Complex16 beta,
                                       ALPHA_Complex16 *y,
                                       ALPHA_Complex16 *d);

/*   Solves triangular system y = alpha * A^{-1} * x   */
alphasparseStatus_t alphasparse_s_trsv(const alphasparseOperation_t operation,
                                      const float alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const float *x,
                                      float *y);

alphasparseStatus_t alphasparse_d_trsv(const alphasparseOperation_t operation,
                                      const double alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const double *x,
                                      double *y);

alphasparseStatus_t alphasparse_c_trsv(const alphasparseOperation_t operation,
                                      const ALPHA_Complex8 alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const ALPHA_Complex8 *x,
                                      ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_trsv(const alphasparseOperation_t operation,
                                      const ALPHA_Complex16 alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const ALPHA_Complex16 *x,
                                      ALPHA_Complex16 *y);

/*   Applies symmetric Gauss-Seidel preconditioner to symmetric system A * x = b, */
/*   that is, it solves:                                                          */
/*      x0       = alpha*x                                                        */
/*      (L+D)*x1 = b - U*x0                                                       */
/*      (D+U)*x  = b - L*x1                                                       */
/*                                                                                */
/*   SYMGS_MV also returns y = A*x                                                */
alphasparseStatus_t alphasparse_s_symgs(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr,
                                       const float alpha,
                                       const float *b,
                                       float *x);

alphasparseStatus_t alphasparse_d_symgs(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr,
                                       const double alpha,
                                       const double *b,
                                       double *x);

alphasparseStatus_t alphasparse_c_symgs(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr,
                                       const ALPHA_Complex8 alpha,
                                       const ALPHA_Complex8 *b,
                                       ALPHA_Complex8 *x);

alphasparseStatus_t alphasparse_z_symgs(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const struct alpha_matrix_descr descr,
                                       const ALPHA_Complex16 alpha,
                                       const ALPHA_Complex16 *b,
                                       ALPHA_Complex16 *x);

alphasparseStatus_t alphasparse_s_symgs_mv(const alphasparseOperation_t op,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr,
                                          const float alpha,
                                          const float *b,
                                          float *x,
                                          float *y);

alphasparseStatus_t alphasparse_d_symgs_mv(const alphasparseOperation_t op,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr,
                                          const double alpha,
                                          const double *b,
                                          double *x,
                                          double *y);

alphasparseStatus_t alphasparse_c_symgs_mv(const alphasparseOperation_t op,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr,
                                          const ALPHA_Complex8 alpha,
                                          const ALPHA_Complex8 *b,
                                          ALPHA_Complex8 *x,
                                          ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_symgs_mv(const alphasparseOperation_t op,
                                          const alphasparse_matrix_t A,
                                          const struct alpha_matrix_descr descr,
                                          const ALPHA_Complex16 alpha,
                                          const ALPHA_Complex16 *b,
                                          ALPHA_Complex16 *x,
                                          ALPHA_Complex16 *y);

/*   Computes an action of a preconditioner
         which corresponds to the approximate matrix decomposition A ≈ (L+D)*E*(U+D)
         for the system Ax = b.

         L is lower triangular part of A
         U is upper triangular part of A
         D is diagonal values of A 
         E is approximate diagonal inverse            
                                                                
         That is, it solves:                                      
             r = rhs - A*x0                                       
             (L + D)*E*(U + D)*dx = r                             
             x1 = x0 + dx                                        */

alphasparseStatus_t alphasparse_s_lu_smoother(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const float *diag,
                                             const float *approx_diag_inverse,
                                             float *x,
                                             const float *rhs);

alphasparseStatus_t alphasparse_d_lu_smoother(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const double *diag,
                                             const double *approx_diag_inverse,
                                             double *x,
                                             const double *rhs);

alphasparseStatus_t alphasparse_c_lu_smoother(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const ALPHA_Complex8 *diag,
                                             const ALPHA_Complex8 *approx_diag_inverse,
                                             ALPHA_Complex8 *x,
                                             const ALPHA_Complex8 *rhs);

alphasparseStatus_t alphasparse_z_lu_smoother(const alphasparseOperation_t op,
                                             const alphasparse_matrix_t A,
                                             const struct alpha_matrix_descr descr,
                                             const ALPHA_Complex16 *diag,
                                             const ALPHA_Complex16 *approx_diag_inverse,
                                             ALPHA_Complex16 *x,
                                             const ALPHA_Complex16 *rhs);

/* Level 3 */

/*   Computes y = alpha * A * x + beta * y   */
alphasparseStatus_t alphasparse_s_mm(const alphasparseOperation_t operation,
                                    const float alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                    const float *x,
                                    const ALPHA_INT columns,
                                    const ALPHA_INT ldx,
                                    const float beta,
                                    float *y,
                                    const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_d_mm(const alphasparseOperation_t operation,
                                    const double alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                    const double *x,
                                    const ALPHA_INT columns,
                                    const ALPHA_INT ldx,
                                    const double beta,
                                    double *y,
                                    const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_c_mm(const alphasparseOperation_t operation,
                                    const ALPHA_Complex8 alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                    const ALPHA_Complex8 *x,
                                    const ALPHA_INT columns,
                                    const ALPHA_INT ldx,
                                    const ALPHA_Complex8 beta,
                                    ALPHA_Complex8 *y,
                                    const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_z_mm(const alphasparseOperation_t operation,
                                    const ALPHA_Complex16 alpha,
                                    const alphasparse_matrix_t A,
                                    const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                    const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                    const ALPHA_Complex16 *x,
                                    const ALPHA_INT columns,
                                    const ALPHA_INT ldx,
                                    const ALPHA_Complex16 beta,
                                    ALPHA_Complex16 *y,
                                    const ALPHA_INT ldy);

/*   Solves triangular system y = alpha * A^{-1} * x   */
alphasparseStatus_t alphasparse_s_trsm(const alphasparseOperation_t operation,
                                      const float alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                      const float *x,
                                      const ALPHA_INT columns,
                                      const ALPHA_INT ldx,
                                      float *y,
                                      const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_d_trsm(const alphasparseOperation_t operation,
                                      const double alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                      const double *x,
                                      const ALPHA_INT columns,
                                      const ALPHA_INT ldx,
                                      double *y,
                                      const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_c_trsm(const alphasparseOperation_t operation,
                                      const ALPHA_Complex8 alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                      const ALPHA_Complex8 *x,
                                      const ALPHA_INT columns,
                                      const ALPHA_INT ldx,
                                      ALPHA_Complex8 *y,
                                      const ALPHA_INT ldy);

alphasparseStatus_t alphasparse_z_trsm(const alphasparseOperation_t operation,
                                      const ALPHA_Complex16 alpha,
                                      const alphasparse_matrix_t A,
                                      const struct alpha_matrix_descr descr, /* alphasparse_matrix_type_t + alphasparse_fill_mode_t + alphasparse_diag_type_t */
                                      const alphasparse_layout_t layout,    /* storage scheme for the dense matrix: C-style or Fortran-style */
                                      const ALPHA_Complex16 *x,
                                      const ALPHA_INT columns,
                                      const ALPHA_INT ldx,
                                      ALPHA_Complex16 *y,
                                      const ALPHA_INT ldy);

/* Sparse-sparse functionality */

/*   Computes sum of sparse matrices: C = alpha * op(A) + B, result is sparse   */
alphasparseStatus_t alphasparse_s_add(const alphasparseOperation_t operation,
                                     const alphasparse_matrix_t A,
                                     const float alpha,
                                     const alphasparse_matrix_t B,
                                     alphasparse_matrix_t *C);

alphasparseStatus_t alphasparse_d_add(const alphasparseOperation_t operation,
                                     const alphasparse_matrix_t A,
                                     const double alpha,
                                     const alphasparse_matrix_t B,
                                     alphasparse_matrix_t *C);

alphasparseStatus_t alphasparse_c_add(const alphasparseOperation_t operation,
                                     const alphasparse_matrix_t A,
                                     const ALPHA_Complex8 alpha,
                                     const alphasparse_matrix_t B,
                                     alphasparse_matrix_t *C);

alphasparseStatus_t alphasparse_z_add(const alphasparseOperation_t operation,
                                     const alphasparse_matrix_t A,
                                     const ALPHA_Complex16 alpha,
                                     const alphasparse_matrix_t B,
                                     alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = op(A) * B, result is sparse   */
alphasparseStatus_t alphasparse_spmm(const alphasparseOperation_t operation,
                                    const alphasparse_matrix_t A,
                                    const alphasparse_matrix_t B,
                                    alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = opA(A) * opB(B), result is sparse   */
alphasparseStatus_t alphasparse_sp2m(const alphasparseOperation_t transA,
                                    const struct alpha_matrix_descr descrA,
                                    const alphasparse_matrix_t A,
                                    const alphasparseOperation_t transB,
                                    const struct alpha_matrix_descr descrB,
                                    const alphasparse_matrix_t B,
                                    const alphasparse_request_t request,
                                    alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is sparse   */
alphasparseStatus_t alphasparse_syrk(const alphasparseOperation_t operation,
                                    const alphasparse_matrix_t A,
                                    alphasparse_matrix_t *C);

/*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is sparse   */
alphasparseStatus_t alphasparse_sypr(const alphasparseOperation_t transA,
                                    const alphasparse_matrix_t A,
                                    const alphasparse_matrix_t B,
                                    const struct alpha_matrix_descr descrB,
                                    alphasparse_matrix_t *C,
                                    const alphasparse_request_t request);

/*   Computes product of sparse matrices: C = op(A) * B * (op(A))^{T for real or H for complex}, result is dense */
alphasparseStatus_t alphasparse_s_syprd(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const float *B,
                                       const alphasparse_layout_t layoutB,
                                       const ALPHA_INT ldb,
                                       const float alpha,
                                       const float beta,
                                       float *C,
                                       const alphasparse_layout_t layoutC,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_syprd(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const double *B,
                                       const alphasparse_layout_t layoutB,
                                       const ALPHA_INT ldb,
                                       const double alpha,
                                       const double beta,
                                       double *C,
                                       const alphasparse_layout_t layoutC,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_syprd(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const ALPHA_Complex8 *B,
                                       const alphasparse_layout_t layoutB,
                                       const ALPHA_INT ldb,
                                       const ALPHA_Complex8 alpha,
                                       const ALPHA_Complex8 beta,
                                       ALPHA_Complex8 *C,
                                       const alphasparse_layout_t layoutC,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_syprd(const alphasparseOperation_t op,
                                       const alphasparse_matrix_t A,
                                       const ALPHA_Complex16 *B,
                                       const alphasparse_layout_t layoutB,
                                       const ALPHA_INT ldb,
                                       const ALPHA_Complex16 alpha,
                                       const ALPHA_Complex16 beta,
                                       ALPHA_Complex16 *C,
                                       const alphasparse_layout_t layoutC,
                                       const ALPHA_INT ldc);

/*   Computes product of sparse matrices: C = op(A) * B, result is dense   */
alphasparseStatus_t alphasparse_s_spmmd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const alphasparse_matrix_t B,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       float *C,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_spmmd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const alphasparse_matrix_t B,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       double *C,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_spmmd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const alphasparse_matrix_t B,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       ALPHA_Complex8 *C,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_spmmd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const alphasparse_matrix_t B,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       ALPHA_Complex16 *C,
                                       const ALPHA_INT ldc);

/*   Computes product of sparse matrices: C = opA(A) * opB(B), result is dense*/
alphasparseStatus_t alphasparse_s_sp2md(const alphasparseOperation_t transA,
                                       const struct alpha_matrix_descr descrA,
                                       const alphasparse_matrix_t A,
                                       const alphasparseOperation_t transB,
                                       const struct alpha_matrix_descr descrB,
                                       const alphasparse_matrix_t B,
                                       const float alpha,
                                       const float beta,
                                       float *C,
                                       const alphasparse_layout_t layout,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_sp2md(const alphasparseOperation_t transA,
                                       const struct alpha_matrix_descr descrA,
                                       const alphasparse_matrix_t A,
                                       const alphasparseOperation_t transB,
                                       const struct alpha_matrix_descr descrB,
                                       const alphasparse_matrix_t B,
                                       const double alpha,
                                       const double beta,
                                       double *C,
                                       const alphasparse_layout_t layout,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_sp2md(const alphasparseOperation_t transA,
                                       const struct alpha_matrix_descr descrA,
                                       const alphasparse_matrix_t A,
                                       const alphasparseOperation_t transB,
                                       const struct alpha_matrix_descr descrB,
                                       const alphasparse_matrix_t B,
                                       const ALPHA_Complex8 alpha,
                                       const ALPHA_Complex8 beta,
                                       ALPHA_Complex8 *C,
                                       const alphasparse_layout_t layout,
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_sp2md(const alphasparseOperation_t transA,
                                       const struct alpha_matrix_descr descrA,
                                       const alphasparse_matrix_t A,
                                       const alphasparseOperation_t transB,
                                       const struct alpha_matrix_descr descrB,
                                       const alphasparse_matrix_t B,
                                       const ALPHA_Complex16 alpha,
                                       const ALPHA_Complex16 beta,
                                       ALPHA_Complex16 *C,
                                       const alphasparse_layout_t layout,
                                       const ALPHA_INT ldc);

/*   Computes product of sparse matrices: C = op(A) * (op(A))^{T for real or H for complex}, result is dense */
alphasparseStatus_t alphasparse_s_syrkd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const float alpha,
                                       const float beta,
                                       float *C,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_d_syrkd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const double alpha,
                                       const double beta,
                                       double *C,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_c_syrkd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const ALPHA_Complex8 alpha,
                                       const ALPHA_Complex8 beta,
                                       ALPHA_Complex8 *C,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_z_syrkd(const alphasparseOperation_t operation,
                                       const alphasparse_matrix_t A,
                                       const ALPHA_Complex16 alpha,
                                       const ALPHA_Complex16 beta,
                                       ALPHA_Complex16 *C,
                                       const alphasparse_layout_t layout, /* storage scheme for the output dense matrix: C-style or Fortran-style */
                                       const ALPHA_INT ldc);

alphasparseStatus_t alphasparse_s_axpy(const ALPHA_INT nz,
                                      const float a,
                                      const float *x,
                                      const ALPHA_INT *indx,
                                      float *y);

alphasparseStatus_t alphasparse_d_axpy(const ALPHA_INT nz,
                                      const double a,
                                      const double *x,
                                      const ALPHA_INT *indx,
                                      double *y);

alphasparseStatus_t alphasparse_c_axpy(const ALPHA_INT nz,
                                      const ALPHA_Complex8 a,
                                      const ALPHA_Complex8 *x,
                                      const ALPHA_INT *indx,
                                      ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_axpy(const ALPHA_INT nz,
                                      const ALPHA_Complex16 a,
                                      const ALPHA_Complex16 *x,
                                      const ALPHA_INT *indx,
                                      ALPHA_Complex16 *y);

alphasparseStatus_t alphasparse_s_gthr(const ALPHA_INT nz,
                                      const float *y,
                                      float *x,
                                      const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_d_gthr(const ALPHA_INT nz,
                                      const double *y,
                                      double *x,
                                      const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_c_gthr(const ALPHA_INT nz,
                                      const ALPHA_Complex8 *y,
                                      ALPHA_Complex8 *x,
                                      const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_z_gthr(const ALPHA_INT nz,
                                      const ALPHA_Complex16 *y,
                                      ALPHA_Complex16 *x,
                                      const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_s_gthrz(const ALPHA_INT nz,
                                       float *y,
                                       float *x,
                                       const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_d_gthrz(const ALPHA_INT nz,
                                       double *y,
                                       double *x,
                                       const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_c_gthrz(const ALPHA_INT nz,
                                       ALPHA_Complex8 *y,
                                       ALPHA_Complex8 *x,
                                       const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_z_gthrz(const ALPHA_INT nz,
                                       ALPHA_Complex16 *y,
                                       ALPHA_Complex16 *x,
                                       const ALPHA_INT *indx);

alphasparseStatus_t alphasparse_s_rot(const ALPHA_INT nz,
                                     float *x,
                                     const ALPHA_INT *indx,
                                     float *y,
                                     const float c,
                                     const float s);

alphasparseStatus_t alphasparse_d_rot(const ALPHA_INT nz,
                                     double *x,
                                     const ALPHA_INT *indx,
                                     double *y,
                                     const double c,
                                     const double s);

alphasparseStatus_t alphasparse_s_sctr(const ALPHA_INT nz,
                                      const float *x,
                                      const ALPHA_INT *indx,
                                      float *y);

alphasparseStatus_t alphasparse_d_sctr(const ALPHA_INT nz,
                                      const double *x,
                                      const ALPHA_INT *indx,
                                      double *y);

alphasparseStatus_t alphasparse_c_sctr(const ALPHA_INT nz,
                                      const ALPHA_Complex8 *x,
                                      const ALPHA_INT *indx,
                                      ALPHA_Complex8 *y);

alphasparseStatus_t alphasparse_z_sctr(const ALPHA_INT nz,
                                      const ALPHA_Complex16 *x,
                                      const ALPHA_INT *indx,
                                      ALPHA_Complex16 *y);

float alphasparse_s_doti(const ALPHA_INT nz,
                        const float *x,
                        const ALPHA_INT *indx,
                        const float *y);

double alphasparse_d_doti(const ALPHA_INT nz,
                         const double *x,
                         const ALPHA_INT *indx,
                         const double *y);

void alphasparse_c_dotci_sub(const ALPHA_INT nz,
                            const ALPHA_Complex8 *x,
                            const ALPHA_INT *indx,
                            const ALPHA_Complex8 *y,
                            ALPHA_Complex8 *dutci);

void alphasparse_z_dotci_sub(const ALPHA_INT nz,
                            const ALPHA_Complex16 *x,
                            const ALPHA_INT *indx,
                            const ALPHA_Complex16 *y,
                            ALPHA_Complex16 *dutci);

void alphasparse_c_dotui_sub(const ALPHA_INT nz,
                            const ALPHA_Complex8 *x,
                            const ALPHA_INT *indx,
                            const ALPHA_Complex8 *y,
                            ALPHA_Complex8 *dutui);

void alphasparse_z_dotui_sub(const ALPHA_INT nz,
                            const ALPHA_Complex16 *x,
                            const ALPHA_INT *indx,
                            const ALPHA_Complex16 *y,
                            ALPHA_Complex16 *dutui);