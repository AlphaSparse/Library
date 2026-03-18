#include <alphasparse/opt.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "alphasparse/format.h"
#include "alphasparse/util.h"
#include "convert_csr_coo.hpp"
#include "transpose_csr.hpp"

template <typename I, typename J, typename T = _internal_spmat>
alphasparseStatus_t convert_csc_csr(const T *source, T **dest){
  T *mat = (T*)alpha_malloc(sizeof(T));
  T *mat_csc = (T*)alpha_malloc(sizeof(T));

  *dest = mat_csc;
  alphasparseStatus_t st;
  st = transpose_csr<J>(source, (T*)&mat);

  mat_csc->rows = source->rows;
  mat_csc->cols = source->cols;
  mat_csc->col_data = mat->row_data;
  // mat_csc->cols_end = mat->rows_end;
  mat_csc->row_data = mat->col_data;
  mat_csc->val_data = mat->val_data;

  // mat_csc->d_cols_end = NULL;
  // mat_csc->d_cols_start = NULL;
  // mat_csc->d_row_indx = NULL;
  // mat_csc->d_values = NULL;
  
  return st;
}
