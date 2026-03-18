#include "test_common.h"
/**
 * @brief ict trsv csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <math.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_trsv(const int argc, const char *argv[], const char *file,
                     int thread_num, const MKL_Complex16 alpha,
                     MKL_Complex16 **ret, size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex16 *values;
  mkl_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  for (MKL_INT i = 0; i < nnz; i++) {
    if (row_index[i] == col_index[i]) {
      values[i].real += 1.0;
      values[i].imag += 1.0;
    }
  }
  size_t size_x = k;
  size_t size_y = m;
  MKL_Complex16 *x =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex16 *y =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_y, DEFAULT_ALIGNMENT);
  alpha_fill_random_z((ALPHA_Complex16 *)x, 1, size_x);

  mkl_set_num_threads(thread_num);

  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, csrA;
  mkl_sparse_z_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_z_trsv(transA, alpha, csrA, descr, x, y),
                "mkl_sparse_z_trsv");
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

static void alpha_trsv(const int argc, const char *argv[], const char *file,
                     int thread_num, const ALPHA_Complex16 alpha,
                     ALPHA_Complex16 **ret, size_t *size, ALPHA_Complex16 **ret_x,
                     size_t *ret_size_x) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  size_t size_x = k;
  size_t size_y = m;
  ALPHA_Complex16 *x =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_x, DEFAULT_ALIGNMENT);
  ALPHA_Complex16 *y =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_y, DEFAULT_ALIGNMENT);
  for (ALPHA_INT i = 0; i < nnz; i++) {
    if (row_index[i] == col_index[i]) {
      values[i].real += 1.0;
      values[i].imag += 1.0;
    }
  }
  alpha_fill_random_z(x, 1, size_x);

  alpha_set_thread_num(thread_num);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, csrA;
  alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(
      alphasparse_convert_csr(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
      "alphasparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_z_trsv_plain(transA, alpha, csrA, descr, x, y),
                "alphasparse_z_trsv_plain");

  alpha_timing_end(&timer);
  printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
         alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));
  alphasparse_destroy(cooA);
  alphasparse_destroy(csrA);

  *ret = y;
  *size = size_y;
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

  const ALPHA_Complex16 alpha_alpha = {.real = 2.0f, .imag = 2.0f};
  const MKL_Complex16 mkl_alpha = {.real = 2.0f, .imag = 2.0f};

  printf("%d,", thread_num);
  ALPHA_Complex16 alpha = {0, 0};
  ALPHA_Complex16 beta = {0, 0};
  ALPHA_Complex16 *alpha_y, *alpha_x;
  MKL_Complex16 *mkl_y;
  size_t size_alpha_y, size_mkl_y;
  size_t size_alpha_x;

  alpha_trsv(argc, argv, file, thread_num, alpha_alpha, &alpha_y, &size_alpha_y, &alpha_x,
           &size_alpha_x);
  int status = 0;
  if (check) {
    alpha_clear_cache();
    printf(",");
    mkl_trsv(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
    printf(",");
    //        status = check_d(double *)mkl_y, size_mkl_y * 2, (double *)alpha_y,
    //        size_alpha_y * 2);
    status = check_z_l2((ALPHA_Complex16 *)mkl_y, size_mkl_y, alpha_y, size_alpha_y,
                        alpha_x, NULL, alpha, beta, argc, argv);
    alpha_free(mkl_y);
  }
  printf("\n");
  alpha_free(alpha_y);

  return status;
}