#include "test_common.h"
/**
 * @brief ict trsm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#ifdef __MKL__
#include <mkl.h>
#endif
#include <stdio.h>
#ifdef __MKL__
static void mkl_trsm(const int argc, const char *argv[], const char *file,
                     int thread_num, const double alpha, double **ret,
                     size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  double *values;
  mkl_set_num_threads(thread_num);
  alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  MKL_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  double *x = (double *)alpha_malloc(size_x * sizeof(double));
  double *y = (double *)alpha_malloc(size_y * sizeof(double));
  alpha_fill_random_d((double *)x, 1, size_x);

  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  int ldx = columns, ldy = columns;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
  }

  sparse_matrix_t cooA, csrA;
  mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_d_trsm(transA, alpha, csrA, descr, layout, x,
                                  columns, ldx, y, ldy),
                "mkl_sparse_d_trsm");
  alpha_timing_end(&timer);
  // printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
  //        alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));
  printf("%s time : %lf[ms]\n", "mkl_sparse_sm",
         alpha_timing_elapsed_time(&timer) * 1000);

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(csrA);

  *ret = y;
  *size = size_y;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
#endif
static void alpha_trsm(const int argc, const char *argv[], const char *file,
                       int thread_num, const double alpha, double **ret_x,
                       size_t *ret_size_x, ALPHA_INT *ret_ldx, double **ret,
                       size_t *size, ALPHA_INT *ret_ldc) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  double *values;
  alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  ALPHA_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  double *x = (double *)alpha_malloc(size_x * sizeof(double));
  double *y = (double *)alpha_malloc(size_y * sizeof(double));
  alpha_fill_random_d(x, 1, size_x);

  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  int ldx = columns, ldy = columns;
  if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
  }
  alpha_set_thread_num(thread_num);

  alphasparse_matrix_t cooA, csrA;
  alpha_call_exit(
      alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                               row_index, col_index, values),
      "alphasparse_d_create_coo");
  alpha_call_exit(alphasparse_convert_csr(
                      cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
                  "alphasparse_convert_csr");
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_call_exit(alphasparse_d_trsm(transA, alpha, csrA, descr, layout, x,
                                     columns, ldx, y, ldy),
                  "alphasparse_d_trsm");
  alpha_timing_end(&timer);
  printf("%s time : %lf[ms]\n", "alphasparse_sm",
         alpha_timing_elapsed_time(&timer) * 1000);

  alphasparse_destroy(cooA);
  alphasparse_destroy(csrA);

  *ret_x = x;
  *ret_ldx = ldx;
  *ret_size_x = size_x;

  *ret = y;
  *ret_ldc = ldy;
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
  bool check = args_get_if_check(argc, argv);

  const double alpha_alpha = 2.f;
  printf("thread num %d\n", thread_num);
#ifdef __MKL__
  const double mkl_alpha = 2.f;
  double *mkl_y;
  size_t size_mkl_y;
#endif
  double *alpha_y, *alpha_x;
  size_t size_alpha_y, size_alpha_x;

  ALPHA_INT ldc, ldx;

  int status = 0;
  alpha_trsm(argc, argv, file, thread_num, alpha_alpha, &alpha_x, &size_alpha_x,
             &ldx, &alpha_y, &size_alpha_y, &ldc);

  if (check) {
#ifdef __MKL__
    mkl_trsm(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
    // printf("size_mkl_y %d,size_alpha_y %d\n",size_mkl_y,size_alpha_y);
    status =
        check_d((double *)mkl_y, size_mkl_y, (double *)alpha_y, size_alpha_y);

    alpha_free(mkl_y);
#endif
  }
  printf("\n");
  alpha_free(alpha_y);
  return status;
}