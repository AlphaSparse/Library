#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "alphasparse/format.h"

template <typename I, typename J, typename T>
alphasparseStatus_t convert_dia_coo(const T *source, T **dest) {
  T *mat = (T*)alpha_malloc(sizeof(T));
  *dest = mat;
  ALPHA_INT rows = source->rows;
  ALPHA_INT cols = source->cols;
  ALPHA_INT nnz = source->nnz;
  ALPHA_INT diag_num = rows + cols - 1;
  bool *flag = (bool *)alpha_malloc((uint64_t)sizeof(bool) * diag_num);
  memset(flag, '\0', (uint64_t)sizeof(bool) * diag_num);
  for (ALPHA_INT i = 0; i < nnz; i++) {
    ALPHA_INT row = source->row_data[i];
    ALPHA_INT col = source->col_data[i];
    ALPHA_INT diag = col - row + rows - 1;
    flag[diag] = 1;
  }
  mat->rows = rows;
  mat->cols = cols;
  mat->lval = rows;
  mat->ndiag = 0;
  for (ALPHA_INT i = 0; i < diag_num; ++i) {
    if (flag[i] == 1) {
      mat->ndiag += 1;
    }
  }
  mat->dis_data = (ALPHA_INT *)alpha_malloc((uint64_t)sizeof(ALPHA_INT) * mat->ndiag);
  for (ALPHA_INT i = 0, index = 0; i < diag_num; ++i) {
    if (flag[i] == 1) {
      mat->dis_data[index] = i - rows + 1;
      index += 1;
    }
  }
  alpha_free(flag);
  ALPHA_INT *diag_pos_map = (ALPHA_INT *)alpha_malloc((uint64_t)sizeof(ALPHA_INT) * diag_num);
  ;
  for (ALPHA_INT i = 0; i < diag_num; i++) {
    diag_pos_map[i] = -1;
  }
  for (ALPHA_INT i = 0; i < mat->ndiag; i++) {
    diag_pos_map[mat->dis_data[i] + rows - 1] = i;
  }
  if ((uint64_t)mat->ndiag * mat->lval > (1l << 31)) {
    fprintf(stderr, "nnz nums %ld overflow!!! \n", (uint64_t)mat->ndiag * mat->lval);
    exit(EXIT_FAILURE);
  }
  mat->val_data = (J *)alpha_malloc((uint64_t)sizeof(J) * mat->ndiag * mat->lval);
  memset(mat->val_data, '\0', (uint64_t)sizeof(J) * mat->ndiag * mat->lval);
  for (ALPHA_INT i = 0; i < nnz; i++) {
    ALPHA_INT row = source->row_data[i];
    ALPHA_INT col = source->col_data[i];
    ALPHA_INT diag = col - row + rows - 1;
    ALPHA_INT pos = diag_pos_map[diag];
    ((J*)mat->val_data)[index2(pos, row, mat->lval)] = ((J*)source->val_data)[i];
  }
  alpha_free(diag_pos_map);

  // mat->d_dis_data = NULL;
  // mat->d_val_data   = NULL;

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
