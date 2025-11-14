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
alphasparseStatus_t create_csc(
    alphasparse_matrix_t *A,
    const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
    const I rows, const I cols, I *cols_start, I *cols_end,
    I *row_indx, J *values) {
  alphasparse_matrix *AA = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
  *A = AA;
  internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(struct _internal_spmat));
  AA->format = ALPHA_SPARSE_FORMAT_CSC;
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
  I nnz = cols_end[cols - 1];
  mat->rows = rows;
  mat->cols = cols;
  I *cols_offset = (I*)alpha_memalign((uint64_t)(cols + 1) * sizeof(I), DEFAULT_ALIGNMENT);
  mat->row_data = (I*)alpha_memalign((uint64_t)nnz * sizeof(I), DEFAULT_ALIGNMENT);
  mat->val_data = alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);
  mat->col_data = cols_offset;
  I* mat_cols_end = cols_offset + 1;
  mat->ordered = false;
  if (indexing == ALPHA_SPARSE_INDEX_BASE_ZERO) {
    cols_offset[0] = cols_start[0];
    for (I i = 0; i < rows; i++) {
      mat_cols_end[i] = cols_end[i];
    }
    for (I i = 0; i < nnz; i++) {
      mat->row_data[i] = row_indx[i];
      ((J *)mat->val_data)[i] = values[i];
    }
  } else {
    cols_offset[0] = cols_start[0] - 1;
    for (I i = 0; i < rows; i++) {
      mat_cols_end[i] = cols_end[i] - 1;
    }
    for (I i = 0; i < nnz; i++) {
      mat->row_data[i] = row_indx[i] - 1;
      ((J *)mat->val_data)[i] = values[i];
    }
  }

  // mat->d_cols_end = NULL;
  // mat->d_cols_start = NULL;
  // mat->d_row_indx = NULL;
  // mat->d_values = NULL;

  // // for dcu
  // AA->dcu_info = NULL;

  return ALPHA_SPARSE_STATUS_SUCCESS;
}