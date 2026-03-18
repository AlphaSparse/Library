#include <alphasparse/opt.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "alphasparse/format.h"
#include "alphasparse/util.h"
#include "alphasparse/util/prefix_sum.h"
#include "convert_csr_coo.hpp"

template <typename I, typename J, typename T = _internal_spmat>
alphasparseStatus_t convert_csc_coo(const T *source, T **dest){
  T *mat = (T*)alpha_malloc(sizeof(T));
  *dest = mat;
  ALPHA_INT m = source->rows;
  ALPHA_INT n = source->cols;
  ALPHA_INT nnz = source->nnz;
  ALPHA_INT num_threads = alpha_get_thread_num();
  mat->rows = m;
  mat->cols = n;
  ALPHA_INT *cols_offset = (ALPHA_INT *)alpha_memalign((uint64_t)(n + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
  mat->row_data = (ALPHA_INT *)alpha_memalign((uint64_t)nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
  mat->val_data = (J *)alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);
  mat->col_data = cols_offset;
  // mat->cols_end = cols_offset + 1;
  memset(cols_offset, 0, (uint64_t)(n + 1) * sizeof(ALPHA_INT));
  mat->col_data[0] = 0;
  ALPHA_INT index = 0;
  // count nnz for each column
  for (ALPHA_INT i = 0; i < nnz; i++) {
    ALPHA_INT col = source->col_data[i];
    mat->col_data[col+1]++;
  }

  prefix_sum(EXL_SCAN, mat->col_data+1, n, cols_offset + 1);

  ALPHA_INT partition[num_threads + 1];
  ALPHA_INT i = 0;

  for (; i < nnz; i++) {
    ALPHA_INT col = source->col_data[i];
    mat->row_data[mat->col_data[col+1]] = source->row_data[i];
    ((J*)mat->val_data)[mat->col_data[col+1]++] = ((J*)source->val_data)[i];
  }
  balanced_partition_row_by_nnz(mat->col_data+1, mat->cols, num_threads, partition);
  // TODO remove following sort code?
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT lrs = partition[tid];
    ALPHA_INT lrh = partition[tid + 1];
    for (ALPHA_INT ar = lrs; ar < lrh; ar++) {
      J *val = &((J*)mat->val_data)[mat->col_data[ar]];
      ALPHA_INT *row_idx = &mat->row_data[mat->col_data[ar]];
      qsort_csr_struct(row_idx, val, 0, mat->col_data[ar+1] - mat->col_data[ar]);
    }
  }

  mat->ordered = true;

  // mat->d_cols_end = NULL;
  // mat->d_cols_start = NULL;
  // mat->d_row_data = NULL;
  // mat->d_val_data = NULL;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
