#include "test_common.h"
/**
 * @brief ict trsv csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_trsv(const int argc, const char *argv[], const char *file,
                     int thread_num, const float alpha, float **ret,
                     size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  float *values;
  mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  size_t size_x = k;
  size_t size_y = m;
  float *x = (float*)alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
  float *y = (float*)alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);
  alpha_fill_random_s(x, 1, size_x);

  for (int i = 0; i < nnz; i++) {
    if (row_index[i] == col_index[i]) {
      values[i] += 1.0;
    }
  }

  mkl_set_num_threads(thread_num);

  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, csrA;
  mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_s_trsv(transA, alpha, csrA, descr, x, y),
                "mkl_sparse_s_trsv");
  alpha_timing_end(&timer);
  printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
         alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));
  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(csrA);

  *ret = y;
  *size = size_y;
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

  printf("%d,", thread_num);

  float *mkl_y;
  size_t size_mkl_y;

  mkl_trsv(argc, argv, file, thread_num, alpha, &mkl_y, &size_mkl_y);
  alpha_free(mkl_y);
  printf("\n");

  return 0;
}