#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

#include <stdio.h>

template <typename TYPE>
static int row_first_cmp(const point_t<TYPE> *a, const point_t<TYPE> *b)
{
  if (a->x != b->x)
    return a->x - b->x;
  return a->y - b->y;
}

template <typename TYPE>
static int col_first_cmp(const point_t<TYPE> *a, const point_t<TYPE> *b)
{
  if (a->y != b->y)
    return a->y - b->y;
  return a->x - b->x;
}

template <typename I, typename J, typename T = _internal_spmat>
alphasparseStatus_t convert_sky_coo(const T *source, T **dest, const alphasparse_fill_mode_t fill)
{
  T *mat = (T *)alpha_malloc(sizeof(T));
  *dest = mat;
  mat->fill = fill;
  mat->rows = source->rows;
  mat->cols = source->cols;
  ALPHA_INT nnz = source->nnz;
  ALPHA_INT m = source->rows;
  ALPHA_INT n = source->cols;
  if (fill == ALPHA_SPARSE_FILL_MODE_LOWER)
  {
    // sort by (row,col)
    point_t<J> *points = (point_t<J> *)alpha_malloc((uint64_t)sizeof(point_t<J>) * nnz);
    ALPHA_INT count = 0;
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
      if (source->row_data[i] >= source->col_data[i])
      {
        points[count].x = source->row_data[i];
        points[count].y = source->col_data[i];
        points[count].v = ((J*)source->val_data)[i];
        count += 1;
      }
    }
    qsort(points, count, sizeof(point_t<J>), (__compar_fn_t)row_first_cmp<J>);
    ALPHA_INT *rows_offset = (ALPHA_INT *)alpha_memalign((uint64_t)(m + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);

    mat->pointers = rows_offset;
    mat->pointers[0] = 0;
    ALPHA_INT64 sky_nnz_row = 0;
    for (ALPHA_INT r = 0, idx = 0; r < mat->rows; ++r)
    {
      ALPHA_INT nnz_row = 0;
      while (idx < count && points[idx].x < r)
        ++idx;
      if(idx == count){ // 已遍历完所有非零元
          mat->pointers[r+1] = mat->pointers[r]+1;
          continue;
      }
      if(points[idx].x == r){
        sky_nnz_row += r - points[idx].y + 1;
        nnz_row = r - points[idx].y + 1;
      }
      else{
        sky_nnz_row += 1;
        nnz_row = 1;
      }
      if(sky_nnz_row > (1l<<31)){

        printf("sky row_offset overflow!!!\n");
        return ALPHA_SPARSE_STATUS_EXECUTION_FAILED;
      }
      mat->pointers[r+1] = mat->pointers[r] + nnz_row;
    }
    ALPHA_INT sky_nnz = mat->pointers[m];
    mat->val_data = (J *)alpha_memalign((uint64_t)sky_nnz * sizeof(J), DEFAULT_ALIGNMENT);
    memset(mat->val_data, '\0', (uint64_t)sky_nnz * sizeof(J));
    // printf("%s:%d sky_nnz is %d\n", __FILE__, __LINE__, sky_nnz);
    for (ALPHA_INT i = 0; i < count; ++i)
    {
      ALPHA_INT row = points[i].x;
      ALPHA_INT col = points[i].y;
      ALPHA_INT row_end = mat->pointers[row + 1];
      ((J*)mat->val_data)[row_end-(row - col + 1)] = points[i].v;
    }
    // printf("%s:%d sky_nnz is %d\n", __FILE__, __LINE__, sky_nnz);
    alpha_free(points);
  }
  else if (fill == ALPHA_SPARSE_FILL_MODE_UPPER)
  {
    // sort by (col,row)
    point_t<J> *points = (point_t<J> *)alpha_malloc((uint64_t)sizeof(point_t<J>) * nnz);
    ALPHA_INT count = 0;
    for (ALPHA_INT i = 0; i < nnz; i++)
    {
      if (source->row_data[i] <= source->col_data[i])
      {
        points[count].x = source->row_data[i];
        points[count].y = source->col_data[i];
        points[count].v = ((J*)source->val_data)[i];
        count += 1;
      }
    }
    qsort(points, count, sizeof(point_t<J>), (__compar_fn_t)col_first_cmp<J>);
    ALPHA_INT *cols_offset = (ALPHA_INT *)alpha_memalign((uint64_t)(n + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->pointers = cols_offset;
    mat->pointers[0] = 0;
    ALPHA_INT64 sky_nnz_row = 0;
    for (ALPHA_INT c = 0, idx = 0; c < mat->cols; ++c)
    {
      while (idx < count && points[idx].y < c)
        ++idx;
      ALPHA_INT col_start = idx;
      if (idx == count)
      { // 已遍历完所有非零元
        sky_nnz_row += 1;
        mat->pointers[c + 1] = mat->pointers[c] + 1;
        continue;
      }
      if (points[idx].y == c)
      {
        sky_nnz_row += c - points[idx].x + 1;
        mat->pointers[c + 1] = mat->pointers[c] + c - points[idx].x + 1;
      }
      else
      { // 空行
        mat->pointers[c + 1] = mat->pointers[c] + 1;
        sky_nnz_row += 1;
      }
      if(sky_nnz_row > (1l<<31)){

        fprintf(stderr,"sky row_offset overflow!!!\n");
        return ALPHA_SPARSE_STATUS_EXECUTION_FAILED;
      }
    }
    ALPHA_INT sky_nnz = mat->pointers[n];
    mat->val_data = (J *)alpha_memalign((uint64_t)sky_nnz * sizeof(J), DEFAULT_ALIGNMENT);
    memset(mat->val_data, '\0', (uint64_t)sky_nnz * sizeof(J));
    for (ALPHA_INT i = 0; i < count; ++i)
    {
      ALPHA_INT row = points[i].x;
      ALPHA_INT col = points[i].y;
      ALPHA_INT col_end = mat->pointers[col + 1];
      ((J*)mat->val_data)[col_end - (col - row + 1)] = points[i].v;
    }
    alpha_free(points);
  }
  else
  {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
