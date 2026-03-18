#include "test_common.h"
/**
 * @brief ict mm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static sparse_status_t alpha_convert_mkl_csc_c(alphasparse_matrix_t src,
                                             sparse_matrix_t *dst,
                                             sparse_index_base_t base,
                                             ALPHA_INT nnz) {
  spmat_csc_z_t *mat = (spmat_csc_z_t *)src->mat;
  sparse_status_t st = mkl_sparse_z_create_csc(
      dst, base, mat->rows, mat->cols, mat->cols_start, mat->cols_end,
      mat->row_indx, (MKL_Complex16 *)mat->values);
  return st;
}

static void mkl_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, MKL_Complex16 alpha, MKL_Complex16 beta,
                   MKL_Complex16 **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

  MKL_INT columns = args_get_columns(argc, argv, k);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_index_base_t base_mkl = SPARSE_INDEX_BASE_ZERO;
  alphasparseIndexBase_t base_ict = ALPHA_SPARSE_INDEX_BASE_ZERO;
  size_t size_x, size_y;
  MKL_INT ldx = columns, ldy = columns;
  if (transA == SPARSE_OPERATION_NON_TRANSPOSE) {
    size_x = k * columns;
    size_y = m * columns;
    if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
      ldx = k;
      ldy = m;
      base_mkl = SPARSE_INDEX_BASE_ONE;
      base_ict = ALPHA_SPARSE_INDEX_BASE_ONE;
    }
  } else {
    size_x = m * columns;
    size_y = k * columns;
    if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
      ldx = m;  // 这里没想明白，为什么不用和NON_TRANS的情况相反
      ldy = k;
      base_mkl = SPARSE_INDEX_BASE_ONE;
      base_ict = ALPHA_SPARSE_INDEX_BASE_ONE;
    }
  }

  MKL_Complex16 *x =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex16 *y =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz * 2);
  alpha_fill_random_d((double *)x, 1, size_x * 2);
  alpha_fill_random_d((double *)y, 1, size_y * 2);

  mkl_set_num_threads(thread_num);
  // sparse_matrix_t coo, csc;
  // mkl_call_exit(mkl_sparse_z_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k,
  // nnz, row_index, col_index, values), "mkl_sparse_z_create_coo");
  // mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE,
  // &csc), "mkl_sparse_convert_csc");
  alphasparse_matrix_t cooA, alpha_cscA;
  sparse_matrix_t cscA;
  alphasparse_z_create_coo(&cooA, base_ict, m, k, nnz,
                          row_index, col_index, values);
  alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
  alpha_convert_mkl_csc_c(alpha_cscA, &cscA, base_mkl, nnz);

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_call_exit(mkl_sparse_z_mm(transA, alpha, cscA, descr, layout, x, columns,
                                ldx, beta, y, ldy),
                "mkl_sparse_z_mm");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_z_mm");

  *ret_y = y;
  *ret_size_y = size_y;

  alphasparse_destroy(cooA);
  alphasparse_destroy(alpha_cscA);
  mkl_sparse_destroy(cscA);
  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

// static void alpha_mm(const int argc, const char *argv[], const char *file, int
// thread_num, ALPHA_Complex16 alpha, ALPHA_Complex16 beta, ALPHA_Complex16 **ret_y,
// size_t *ret_size_y)
static void alpha_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, ALPHA_Complex16 alpha, ALPHA_Complex16 beta,
                   ALPHA_Complex16 **ret_x, size_t *ret_size_x, ALPHA_INT *ret_ldx,
                   ALPHA_Complex16 **ret_y, size_t *ret_size_y,
                   ALPHA_INT *ret_ldy) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

  ALPHA_INT columns = args_get_columns(argc, argv, k);
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  size_t size_x, size_y;
  ALPHA_INT ldx = columns, ldy = columns;
  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    size_x = k * columns;
    size_y = m * columns;
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
      ldx = k;
      ldy = m;
    }
  } else {
    size_x = m * columns;
    size_y = k * columns;
    if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
      ldx = m;
      ldy = k;
    }
  }

  ALPHA_Complex16 *x =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_x, DEFAULT_ALIGNMENT);
  ALPHA_Complex16 *y =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_z(values, 1, nnz);
  alpha_fill_random_z(x, 1, size_x);
  alpha_fill_random_z(y, 1, size_y);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, csc;
  alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csc),
      "alphasparse_convert_csc");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_z_mm_plain(transA, alpha, csc, descr, layout, x,
                                      columns, ldx, beta, y, ldy),
                "alphasparse_z_mm");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_z_mm");

  alphasparse_destroy(coo);
  alphasparse_destroy(csc);

  *ret_x = x;
  *ret_ldx = ldx;
  *ret_size_x = size_x;

  *ret_y = y;
  *ret_ldy = ldy;
  *ret_size_y = size_y;

  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  const char *file = args_get_data_file(argc, argv);
  int thread_num = args_get_thread_num(argc, argv);
  bool check = args_get_if_check(argc, argv);

  const ALPHA_Complex16 alpha = {.real = 3., .imag = 3.};
  const ALPHA_Complex16 beta = {.real = 2., .imag = 2.};

  const MKL_Complex16 mkl_alpha = {.real = 3., .imag = 3.};
  const MKL_Complex16 mkl_beta = {.real = 2., .imag = 2.};

  printf("thread_num : %d\n", thread_num);

  ALPHA_Complex16 *alpha_y;
  MKL_Complex16 *mkl_y;
  size_t size_alpha_y, size_mkl_y;
  ALPHA_Complex16 *alpha_x;
  size_t size_alpha_x;
  ALPHA_INT ldx, ldy;

  // alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);
  // alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_x, &size_alpha_x,
  // &ldx, &alpha_y, &size_alpha_y, &ldy);

  int status = 0;
  if (check) {
    mkl_mm(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y,
           &size_mkl_y);
    alpha_free(mkl_y);
  }

  // alpha_free(alpha_y);
  return status;
}