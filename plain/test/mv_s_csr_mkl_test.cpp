#include "test_common.h"
/**
 * @brief ict mv csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_mv(const int argc, const char *argv[], const char *file,
                   int thread_num, const float alpha, const float beta,
                   float **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  float *values;
  mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);
  size_t size_x = k;
  size_t size_y = m;
  if (transA == SPARSE_OPERATION_TRANSPOSE ||
      transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }
  float *x = (float*)alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
  float *y = (float*)alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_s(x, 1, size_x);
  alpha_fill_random_s(y, 1, size_y);

  mkl_set_num_threads(thread_num);

  sparse_matrix_t cooA, csrA;
  mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);

  alpha_clear_cache();

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_sparse_s_mv(transA, alpha, csrA, descr, x, beta, y);

  alpha_timing_end(&timer);

  printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
         alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(csrA);

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

  const float alpha = 2;
  const float beta = 3;

  float *mkl_y;
  size_t size_mkl_y;

  printf("%d,", thread_num);
  mkl_mv(argc, argv, file, thread_num, alpha, beta, &mkl_y, &size_mkl_y);
  alpha_free(mkl_y);
  printf("\n");
  return 0;
}
