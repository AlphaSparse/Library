#include "test_common.h"
/**
 * @brief ict mv csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static sparse_status_t alpha_convert_mkl_csc_c(alphasparse_matrix_t src,
                                             sparse_matrix_t *dst) {
  spmat_csc_c_t *mat = (spmat_csc_c_t *)src->mat;
  sparse_status_t st = mkl_sparse_c_create_csc(
      dst, SPARSE_INDEX_BASE_ZERO, mat->rows, mat->cols, mat->cols_start,
      mat->cols_end, mat->row_indx, (MKL_Complex8 *)mat->values);
  return st;
}

static void mkl_mv(const int argc, const char *argv[], const char *file,
                   int thread_num, const MKL_Complex8 alpha,
                   const MKL_Complex8 beta, MKL_Complex8 **ret_y,
                   size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  ALPHA_Complex8 *values;
  alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_x, size_y;
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  if (transA == SPARSE_OPERATION_NON_TRANSPOSE) {
    size_x = k;
    size_y = m;
  } else {
    // 转置，共轭转置
    size_x = m;
    size_y = k;
  }
  MKL_Complex8 *x =
      (MKL_Complex8*)alpha_memalign(sizeof(MKL_Complex8) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex8 *y =
      (MKL_Complex8*)alpha_memalign(sizeof(MKL_Complex8) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_s((float *)values, 1, nnz * 2);
  alpha_fill_random_s((float *)x, 1, size_x * 2);
  alpha_fill_random_s((float *)y, 1, size_y * 2);

  mkl_set_num_threads(thread_num);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, alpha_cscA;
  sparse_matrix_t cscA;
  alphasparse_c_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                          row_index, col_index, values);
  alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
  alpha_convert_mkl_csc_c(alpha_cscA, &cscA);

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_sparse_c_mv(transA, alpha, cscA, descr, x, beta, y);

  alpha_timing_end(&timer);

  alpha_timing_elaped_time_print(&timer, "mkl_sparse_c_mv");

  mkl_sparse_destroy(cscA);
  alphasparse_destroy(cooA);
  alphasparse_destroy(alpha_cscA);

  *ret_y = y;
  *ret_size_y = size_y;

  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
// static void alpha_mv(const int argc, const char *argv[], const char *file, int
// thread_num, const ALPHA_Complex8 alpha, const ALPHA_Complex8 beta, ALPHA_Complex8
// **ret_y, size_t *ret_size_y)
static void alpha_mv(const int argc, const char *argv[], const char *file,
                   int thread_num, const ALPHA_Complex8 alpha,
                   const ALPHA_Complex8 beta, ALPHA_Complex8 **ret_y,
                   size_t *ret_size_y, ALPHA_Complex8 **ret_x,
                   size_t *ret_size_x) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex8 *values;
  alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
  size_t size_x, size_y;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    size_x = k;
    size_y = m;
  } else {
    // 转置，共轭转置
    size_x = m;
    size_y = k;
  }

  ALPHA_Complex8 *x =
      (ALPHA_Complex8*)alpha_memalign(sizeof(ALPHA_Complex8) * size_x, DEFAULT_ALIGNMENT);
  ALPHA_Complex8 *y =
      (ALPHA_Complex8*)alpha_memalign(sizeof(ALPHA_Complex8) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_c(values, 1, nnz);
  alpha_fill_random_c(x, 1, size_x);
  alpha_fill_random_c(y, 1, size_y);

  alpha_set_thread_num(thread_num);

  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, cscA;
  alpha_call_exit(alphasparse_c_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_c_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA),
      "alphasparse_convert_csc");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_c_mv_plain(transA, alpha, cscA, descr, x, beta, y),
                "alphasparse_c_mv");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_c_mv");
  alphasparse_destroy(cooA);
  alphasparse_destroy(cscA);

  *ret_y = y;
  *ret_size_y = size_y;

  *ret_x = x;
  *ret_size_x = size_x;

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

  const ALPHA_Complex8 alpha = {.real = 3., .imag = 3.};
  const ALPHA_Complex8 beta = {.real = 2., .imag = 2.};

  const MKL_Complex8 mkl_alpha = {.real = 3., .imag = 3.};
  const MKL_Complex8 mkl_beta = {.real = 2., .imag = 2.};

  ALPHA_Complex8 *alpha_y;
  MKL_Complex8 *mkl_y;
  ALPHA_Complex8 *alpha_x;
  MKL_Complex8 *mkl_x;
  size_t size_alpha_y, size_mkl_y;
  size_t size_alpha_x;

  printf("thread_num : %d\n", thread_num);

  // alpha_mv(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);
  // alpha_mv(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y,
  // &alpha_x, &size_alpha_x);
  int status = 0;

  if (check) {
    mkl_mv(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y,
           &size_mkl_y);
    // status = check_s((ALPHA_Complex8 *)mkl_y, size_mkl_y * 2, (ALPHA_Complex8
    // *)alpha_y, size_alpha_y * 2); status = check_c_l2((ALPHA_Complex8 *)mkl_y,
    // size_mkl_y, alpha_y, size_alpha_y, alpha_x, NULL, alpha, beta, argc, argv);
    alpha_free(mkl_y);
  }

  // alpha_free(alpha_y);
  return status;
}