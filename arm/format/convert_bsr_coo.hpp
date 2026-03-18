#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>

#include "alphasparse/format.h"
#include "alphasparse/util/bitmap.h"
#include "convert_csr_coo.hpp"
#include "destroy_csr.hpp"

template <typename TYPE>
struct coord_t{
  ALPHA_INT row_idx;
  ALPHA_INT col_idx;
  TYPE value;
};

#define ROWS_PER_ROUND (4)
#define NNZ_PADDING_RATIO_BOUND (15.0)
template <typename TYPE>
static int cmp_coord(const void *a, const void *b) {
  return ((const coord_t<TYPE> *)a)->col_idx - ((const coord_t<TYPE> *)b)->col_idx;
}
// TODO 不应该依赖排序的CSR
template <typename I, typename J, typename T>
alphasparseStatus_t convert_bsr_coo(const T *source, T **dest, 
                          const ALPHA_INT block_size, const alphasparse_layout_t block_layout) {
  // alpha_timer_t timer;
  ALPHA_INT m = source->rows;
  ALPHA_INT n = source->cols;
  ALPHA_INT nnz = source->nnz;
  if (m % block_size != 0 || n % block_size != 0) {
    printf("in convert_bsr_hints_coo , rows or cols is not divisible by block_size!!!");
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
  T *mat = (T *)alpha_malloc(sizeof(T));
  *dest = mat;
  ALPHA_INT block_rows = m / block_size;
  ALPHA_INT block_cols = n / block_size;
  ALPHA_INT *block_row_offset = (ALPHA_INT *)
      alpha_memalign((uint64_t)(block_rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
  mat->rows = block_rows;
  mat->cols = block_cols;
  mat->block_dim = block_size;
  mat->block_layout = block_layout;
  mat->row_data = block_row_offset;
  ALPHA_INT * rows_end = block_row_offset + 1;
  T *csr;
  check_error_return((convert_csr_coo<I, J, T>(source, &csr)));

  mat->row_data[0] = 0;
  ALPHA_INT num_threads = alpha_get_thread_num();
  ALPHA_INT block_nnz = 0;

  int blk_sft = 0;
  if (((block_size - 1) & block_size) == 0) {
    int bs = block_size >> 1;
    while (bs) {
      bs >>= 1;
      blk_sft++;
    }
  } else {
    printf("block_size is not power of two\n");
    exit(-1);
  }
  // alpha_timing_start(&timer);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    bitmap_t *bitmap;
    bitmap_build(&bitmap, block_cols);
#ifdef _OPENMP
#pragma omp for
#endif
    for (ALPHA_INT row = 0; row < m; row += block_size) {
      const ALPHA_INT start = csr->row_data[row];
      const ALPHA_INT end = csr->row_data[row + block_size];
      ALPHA_INT nz_blk_num =
          set_clear_bit_batch_sht_index(bitmap, &csr->col_data[start], end - start, blk_sft);
      // printf("br %d has %d nnz_block, nnz is %d \n", row >> blk_sft, nz_blk_num,end - start);
      rows_end[row >> blk_sft] = nz_blk_num;
    }
    bitmap_destory(bitmap);
  }

  for (ALPHA_INT br = 0; br < block_rows; br++) {
    rows_end[br] = rows_end[br] + rows_end[br - 1];
  }

  block_nnz = rows_end[block_rows - 1];
  const double nnz_padding = 1.0 * block_nnz * block_size * block_size / nnz;

  mat->col_data = (ALPHA_INT *)alpha_memalign((uint64_t)block_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
  mat->val_data = (J *)alpha_memalign((uint64_t)block_nnz * block_size * block_size * sizeof(J),
                             DEFAULT_ALIGNMENT);
  ALPHA_INT *partition = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (num_threads + 1));
  balanced_partition_row_by_nnz(rows_end, block_rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT lrs = partition[tid];
    ALPHA_INT lrh = partition[tid + 1];
    J *values = &((J*)mat->val_data)[mat->row_data[lrs] * block_size * block_size];
    // count: nnz_block
    ALPHA_INT count = rows_end[lrh - 1] - mat->row_data[lrs];
    memset(values, '\0', (uint64_t)count * block_size * block_size * sizeof(J));
    // alpha_timing_start(&timer);
    for (ALPHA_INT br = lrs; br < lrh; br++) {
      J *values_current_rowblk =
          &((J*)mat->val_data)[mat->row_data[br] * block_size * block_size];
      const ALPHA_INT row_s = br * block_size;
      const ALPHA_INT total_nnz = csr->row_data[row_s + block_size] - csr->row_data[row_s];
      if (total_nnz == 0) {
        continue;
      }
      coord_t<J> *points_current_rowblk = (coord_t<J> *)alpha_malloc(sizeof(coord_t<J>) * total_nnz);
      ALPHA_INT *bsr_col_index = &mat->col_data[mat->row_data[br]];
      // points_current_rowblk 存储原始矩阵的列坐标 / block_size
      for (ALPHA_INT ir = 0, nnz = 0; ir < block_size; ir++) {
        ALPHA_INT r = br * block_size + ir;
        ALPHA_INT start = csr->row_data[r];
        ALPHA_INT end = csr->row_data[r+1];

        for (ALPHA_INT ai = start; ai < end; ai++) {
          points_current_rowblk[nnz].col_idx = csr->col_data[ai];
          points_current_rowblk[nnz].row_idx = r;
          points_current_rowblk[nnz].value = ((J*)csr->val_data)[ai];
          nnz++;
        }
      }

      qsort(points_current_rowblk, total_nnz, sizeof(coord_t<J>), (__compar_fn_t)cmp_coord<J>);

      ALPHA_INT idx = 0;
      bsr_col_index[idx] = points_current_rowblk[0].col_idx / block_size;
      ALPHA_INT pre = points_current_rowblk[0].col_idx / block_size;
      ALPHA_INT ir = points_current_rowblk[0].row_idx % block_size;
      ALPHA_INT ic = points_current_rowblk[0].col_idx % block_size;
      J *values_current_blk = values_current_rowblk;

      if (block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
        values_current_blk[ir * block_size + ic] = points_current_rowblk[0].value;
        //  points_current_rowblk存储每个nnz对应 bsr当前行中具体哪一个块
        for (ALPHA_INT nnz = 1; nnz < total_nnz; nnz++) {
          // next blk
          if (pre != points_current_rowblk[nnz].col_idx / block_size) {
            idx++;
            values_current_blk += block_size * block_size;
            bsr_col_index[idx] = points_current_rowblk[nnz].col_idx / block_size;
          }
          pre = points_current_rowblk[nnz].col_idx / block_size;
          ic = points_current_rowblk[nnz].col_idx % block_size;
          ir = points_current_rowblk[nnz].row_idx % block_size;
          values_current_blk[ir * block_size + ic] = points_current_rowblk[nnz].value;
        }
      } else {
        values_current_blk[ic * block_size + ir] = points_current_rowblk->value;
        //  points_current_rowblk存储每个nnz对应 bsr当前行中具体哪一个块
        for (ALPHA_INT nnz = 1; nnz < total_nnz; nnz++) {
          if (pre != points_current_rowblk[nnz].col_idx / block_size) {
            idx++;
            values_current_blk += block_size * block_size;
            bsr_col_index[idx] = points_current_rowblk[nnz].col_idx / block_size;
          }
          ir = points_current_rowblk[nnz].row_idx % block_size;
          ic = points_current_rowblk[nnz].col_idx % block_size;
          pre = points_current_rowblk[nnz].col_idx / block_size;
          values_current_blk[ic * block_size + ir] = points_current_rowblk[nnz].value;
        }
      }
      const ALPHA_INT block_nnz_br = mat->row_data[br+1] - mat->row_data[br];
      if (idx != block_nnz_br - 1) {
        fprintf(stderr,
                "god, some error occurs, block_nnz of current br %d wrong expected %d, got %d \n",
                br, block_nnz_br, idx + 1);
        exit(-1);
      }
      alpha_free(points_current_rowblk);
    }
  }

  mat->ordered = true;
  destroy_csr(csr);
  alpha_free(partition);

  // mat->d_col_indx = NULL;
  // mat->d_rows_ptr = NULL;
  // mat->d_values   = NULL;
  return ALPHA_SPARSE_STATUS_SUCCESS;
}
