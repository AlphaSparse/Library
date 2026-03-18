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
alphasparseStatus_t create_coo(
    alphasparse_matrix_t *A,
    const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
    const I rows, const I cols, const I nnz, I *row_indx, I *col_indx,
    J *values) {
  alphasparse_matrix *AA = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
  *A = AA;
  internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(struct _internal_spmat));
  AA->mat = mat;
  AA->format = ALPHA_SPARSE_FORMAT_COO;
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
  
  AA->inspector = NULL;
  AA->inspector = (alphasparse_inspector_t)alpha_malloc(sizeof(alphasparse_inspector));
  alphasparse_inspector *kernel_inspector = (alphasparse_inspector *)AA->inspector;
  kernel_inspector->mv_inspector = NULL;
  kernel_inspector->request_kernel = ALPHA_NONE;
  kernel_inspector->mm_inspector = NULL;
  kernel_inspector->mmd_inspector = NULL;
  kernel_inspector->sv_inspector = NULL;
  kernel_inspector->sm_inspector = NULL;
  kernel_inspector->memory_policy = ALPHA_SPARSE_MEMORY_AGGRESSIVE;

  AA->mat->rows = rows;
  AA->mat->cols = cols;
  AA->mat->nnz = nnz;
  AA->mat->row_data = (I*)alpha_memalign((uint64_t)sizeof(I) * nnz, DEFAULT_ALIGNMENT);
  AA->mat->col_data = (I*)alpha_memalign((uint64_t)sizeof(I) * nnz, DEFAULT_ALIGNMENT);
  AA->mat->val_data = alpha_memalign((uint64_t)sizeof(J) * nnz, DEFAULT_ALIGNMENT);
  // AA->mat->ordered = false;
  if (indexing == ALPHA_SPARSE_INDEX_BASE_ZERO) {
    for (I i = 0; i < nnz; ++i) {
      AA->mat->row_data[i] = row_indx[i];
      AA->mat->col_data[i] = col_indx[i];
      ((J*)(AA->mat->val_data))[i] = values[i];
    }
  } else {
    for (I i = 0; i < nnz; ++i) {
      AA->mat->row_data[i] = row_indx[i] - 1;
      AA->mat->col_data[i] = col_indx[i] - 1;
      ((J*)(AA->mat->val_data))[i] = values[i];
    }
  }

#ifdef __DCU__
  coo_order<J>(mat);
#endif

  // for dcu
  // AA->dcu_info     = NULL;
  // AA->mat->d_values    = NULL;
  // AA->mat->d_cols_indx = NULL;
  // AA->mat->d_rows_indx = NULL;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}