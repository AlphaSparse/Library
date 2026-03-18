#include <alphasparse/opt.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "alphasparse/format.h"
#include "alphasparse/util.h"
#include "alphasparse/util/prefix_sum.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/util/partition.h"

template <typename I>
ALPHA_INT qsort_partition_template(ALPHA_INT *col_index, I *values, const ALPHA_INT left, const ALPHA_INT right)
{
    ALPHA_INT select = left + ((right - left) >> 1);

    ALPHA_INT tmp_col;
    I tmp_val;

    tmp_col = col_index[left];
    col_index[left] = col_index[select];
    col_index[select] = tmp_col;

    tmp_val = values[left];
    values[left] = values[select];
    values[select] = tmp_val;

    ALPHA_INT pivot = left ;

    ALPHA_INT col_pivot = col_index[pivot];
    I val_pivot = values[pivot];

    ALPHA_INT p = left + 1;
    ALPHA_INT q = right - 1;
    while (true)
    {
        while (p < right && col_index[p] < col_pivot)
        {
            p++;
        }
        while (q > left && col_index[q] >= col_pivot)
        {
            q--;
        }
        if (p >= q)
            break;

        tmp_col = col_index[p];
        col_index[p] = col_index[q];
        col_index[q] = tmp_col;

        tmp_val = values[p];
        values[p] = values[q];
        values[q] = tmp_val;
    }

    col_index[pivot] = col_index[q];
    col_index[q] = col_pivot;

    values[pivot] = values[q];
    values[q] = val_pivot;

    return q;
}

template <typename TYPE>                                                  
void qsort_csr_struct(ALPHA_INT *col_index, TYPE *values, const ALPHA_INT left, const ALPHA_INT right) 
{                                                                                           
    if(left>=right)                                                                         
        return;                                                                             
    ALPHA_INT med = qsort_partition_template(col_index, values, left, right);               
    qsort_csr_struct(col_index, values, left, med);                                                    
    qsort_csr_struct(col_index, values, med + 1, right);                                               
}

template <typename I, typename J, typename T>
alphasparseStatus_t convert_csr_coo(const T *source, T **dest) {
  T *mat = (T*)alpha_malloc(sizeof(T));
  // alpha_timer_t timer;
  *dest = mat;
  I m = source->rows;
  I n = source->cols;
  I nnz = source->nnz;
  I num_threads = alpha_get_thread_num();
  mat->rows = m;
  mat->cols = n;

  I *rows_offset_scan =
      (I*)alpha_memalign((uint64_t)(m + 1) * sizeof(I), DEFAULT_ALIGNMENT);
  mat->col_data =
      (I*)alpha_memalign((uint64_t)nnz * sizeof(I), DEFAULT_ALIGNMENT);
  mat->val_data =
      (J*)alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);

  I *tmp_col_indx =
      (I*)alpha_memalign((uint64_t)nnz * sizeof(I), DEFAULT_ALIGNMENT);
  J *tmp_values =
      (J*)alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);

  mat->row_data = rows_offset_scan;
  I* rows_end = rows_offset_scan + 1;
  memset(rows_offset_scan, 0, (uint64_t)(m + 1) * sizeof(I));
  mat->row_data[0] = 0;
  I index = 0;
  // alpha_timing_start(&timer);
  // count nnz for each row
  for (I i = 0; i < nnz; i++) {
    I row = source->row_data[i];
    rows_end[row]++;
  }
  // alpha_timing_end(&timer);
  // double total_time = alpha_timing_elapsed_time(&timer) * 1000;
  // printf("count nnz time %f ms\n", total_time);

  prefix_sum(EXL_SCAN, rows_end, m, rows_offset_scan + 1);

  I partition[num_threads + 1];

  {
    I i = 0;
    for (; i < nnz - 3; i += 4) {
      I row0 = source->row_data[i];
      I row1 = source->row_data[i + 1];
      I row2 = source->row_data[i + 2];
      I row3 = source->row_data[i + 3];

      tmp_col_indx[rows_end[row0]] = source->col_data[i + 0];
      tmp_values[rows_end[row0]++] = ((J*)source->val_data)[i + 0];

      tmp_col_indx[rows_end[row1]] = source->col_data[i + 1];
      tmp_values[rows_end[row1]++] = ((J*)source->val_data)[i + 1];

      tmp_col_indx[rows_end[row2]] = source->col_data[i + 2];
      tmp_values[rows_end[row2]++] = ((J*)source->val_data)[i + 2];

      tmp_col_indx[rows_end[row3]] = source->col_data[i + 3];
      tmp_values[rows_end[row3]++] = ((J*)source->val_data)[i + 3];
    }
    for (; i < nnz; i++) {
      I row = source->row_data[i];
      tmp_col_indx[rows_end[row]] = source->col_data[i];
      tmp_values[rows_end[row]++] = ((J*)source->val_data)[i];
    }
  }

  balanced_partition_row_by_nnz(rows_end, mat->rows, num_threads,
                                partition);
// copy data in parallel
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    I tid = alpha_get_thread_id();
    I lrs = partition[tid];
    I lrh = partition[tid + 1];
    for (I ar = lrs; ar < lrh; ar++) {
      J *val = &tmp_values[mat->row_data[ar]];
      I *col_idx = &tmp_col_indx[mat->row_data[ar]];
      qsort_csr_struct(col_idx, val, 0,
                       rows_end[ar] - mat->row_data[ar]);
    }
    memcpy(mat->col_data + mat->row_data[lrs],
           tmp_col_indx + mat->row_data[lrs],
           sizeof(I) * (mat->row_data[lrh] - mat->row_data[lrs]));
    memcpy(((J*)mat->val_data) + mat->row_data[lrs],
           tmp_values + mat->row_data[lrs],
           sizeof(J) * (mat->row_data[lrh] - mat->row_data[lrs]));
  }
  // mat->ordered = true;
  alpha_free(tmp_col_indx);
  alpha_free(tmp_values);
  
  // mat->d_col_indx = NULL;
  // mat->d_row_ptr  = NULL;
  // mat->d_values   = NULL;

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
