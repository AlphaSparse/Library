#include "test_common.h"
/**
 * @brief ict mv sky test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_mv(const int argc, const char *argv[], const char *file,
                   int thread_num, const double alpha, const double beta,
                   double **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  double *values;
  mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_x = k;
  size_t size_y = m;
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d(values, 1, nnz);
  alpha_fill_random_d(x, 1, size_x);
  alpha_fill_random_d(y, 1, size_y);

  mkl_set_num_threads(thread_num);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, skyA;
  mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &skyA);

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_sparse_d_mv(transA, alpha, skyA, descr, x, beta, y);

  alpha_timing_end(&timer);

  alpha_timing_elaped_time_print(&timer, "mkl_sparse_d_mv");

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(skyA);

  *ret_y = y;
  *ret_size_y = size_y;

  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
static void alpha_mv(const int argc, const char *argv[], const char *file,
                   int thread_num, const double alpha, const double beta,
                   double **ret_y, size_t *ret_size_y) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  double *values;
  alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_x = k;
  size_t size_y = m;
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d(values, 1, nnz);
  alpha_fill_random_d(x, 1, size_x);
  alpha_fill_random_d(y, 1, size_y);

  alpha_set_thread_num(thread_num);

  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, skyA;
  alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_d_create_coo");
  alpha_call_exit(alphasparse_convert_sky(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                       descr.mode, &skyA),
                "alphasparse_convert_sky");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_d_mv_plain(transA, alpha, skyA, descr, x, beta, y),
                "alphasparse_d_mv");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_d_mv");
  alphasparse_destroy(cooA);
  alphasparse_destroy(skyA);

  *ret_y = y;
  *ret_size_y = size_y;

  alpha_free(x);
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

  const double alpha = 2;
  const double beta = 3;

  double *alpha_y, *mkl_y;
  size_t size_alpha_y, size_mkl_y;

  printf("thread_num : %d\n", thread_num);

  alpha_mv(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);

  int status = 0;

  if (check) {
    mkl_mv(argc, argv, file, thread_num, alpha, beta, &mkl_y, &size_mkl_y);
    status = check_d(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
    alpha_free(mkl_y);
  }

  alpha_free(alpha_y);
  return status;
}