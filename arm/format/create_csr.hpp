// #include "alphasparse.h"
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <alphasparse/format.h>
#include <alphasparse/spapi.h>

#include <memory.h>
#include <stdlib.h>

#include "alphasparse/inspector.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/malloc.h"
#include "coo_order.hpp"
#include <type_traits>

template <typename I, typename J>
alphasparseStatus_t create_csr(
    alphasparse_matrix_t *A,
    const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
    const I rows, const I cols, I *rows_start, I *rows_end,
    I *col_indx, J *values) {
  alphasparse_matrix *AA = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
  *A = AA;
  internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(struct _internal_spmat));
  AA->format = ALPHA_SPARSE_FORMAT_CSR;
  if(std::is_same_v<J, float>)
    AA->datatype_cpu = ALPHA_SPARSE_DATATYPE_FLOAT;
  else if(std::is_same_v<J, double>)
    AA->datatype_cpu = ALPHA_SPARSE_DATATYPE_DOUBLE;
  else if(std::is_same_v<J, ALPHA_Complex8>)
    AA->datatype_cpu = ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX;
  else if(std::is_same_v<J, ALPHA_Complex16>)
    AA->datatype_cpu = ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX;
  else
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  AA->mat = mat;
  // AA->inspector = NULL;
  // AA->inspector = (alphasparse_inspector_t)alpha_malloc(sizeof(alphasparse_inspector));
  // alphasparse_inspector *kernel_inspector = (alphasparse_inspector *)AA->inspector;
  // kernel_inspector->mv_inspector = NULL;
  // kernel_inspector->request_kernel = ALPHA_NONE;
  // kernel_inspector->mm_inspector = NULL;
  // kernel_inspector->mmd_inspector = NULL;
  // kernel_inspector->sv_inspector = NULL;
  // kernel_inspector->sm_inspector = NULL;
  // kernel_inspector->memory_policy = ALPHA_SPARSE_MEMORY_AGGRESSIVE;
  I nnz = rows_end[rows - 1] - rows_start[0];
  mat->rows = rows;
  mat->cols = cols;
  I *rows_offset = (I*)alpha_memalign((uint64_t)(rows + 1) * sizeof(I), DEFAULT_ALIGNMENT);
  mat->col_data = (I*)alpha_memalign((uint64_t)nnz * sizeof(I), DEFAULT_ALIGNMENT);
  mat->val_data = alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);
  mat->row_data = rows_offset;
  I * mat_rows_end = rows_offset + 1;
  mat->ordered = false;
  if (indexing == ALPHA_SPARSE_INDEX_BASE_ZERO) {
    mat->row_data[0] = rows_start[0];
    for (I i = 0; i < rows; i++) {
      mat_rows_end[i] = rows_end[i];
    }
    for (I i = 0; i < nnz; i++) {
      mat->col_data[i] = col_indx[i];
      ((J *)mat->val_data)[i] = values[i];
    }
  } else {
    mat->row_data[0] = rows_start[0] - 1;
    for (I i = 0; i < rows; i++) {
      mat_rows_end[i] = rows_end[i] - 1;
    }
    for (I i = 0; i < nnz; i++) {
      mat->col_data[i] = col_indx[i] - 1;
      ((J *)mat->val_data)[i] = values[i];
    }
  }

// #ifdef __DCU__
//   csr_order(mat);
// #endif

//   // for dcu
//   mat->d_col_indx  = NULL;
//   mat->d_row_ptr   = NULL;
//   mat->d_values    = NULL;
//   AA->dcu_info     = NULL;

  return ALPHA_SPARSE_STATUS_SUCCESS;
}