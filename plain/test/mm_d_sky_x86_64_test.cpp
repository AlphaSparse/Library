#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"
static void mkl_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, double alpha, double beta, double **ret_y,
                   size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  double *values;
  mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  MKL_INT columns = args_get_columns(argc, argv, k);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);
  MKL_INT rowsx = k, rowsy = m;
  if (transA == SPARSE_OPERATION_TRANSPOSE ||
      transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rowsx = m;
    rowsy = k;
  }
  MKL_INT ldx = columns, ldy = columns;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = rowsx;
    ldy = rowsy;
  }
  size_t size_x = rowsx * columns;
  size_t size_y = rowsy * columns;
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);
  alpha_fill_random_d(x, 1, size_x);
  alpha_fill_random_d(y, 1, size_y);
  mkl_set_num_threads(thread_num);
  sparse_matrix_t coo, csr;
  mkl_call_exit(mkl_sparse_d_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                                        row_index, col_index, values),
                "mkl_sparse_d_create_coo");
  mkl_call_exit(
      mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr),
      "mkl_sparse_convert_csr");
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_d_mm(transA, alpha, csr, descr, layout, x, columns,
                                ldx, beta, y, ldy),
                "mkl_sparse_d_mm");
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_d_mm");
  *ret_y = y;
  *ret_size_y = size_y;
  mkl_sparse_destroy(coo);
  mkl_sparse_destroy(csr);
  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
static void alpha_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, double alpha, double beta, double **ret_y,
                   size_t *ret_size_y) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  double *values;
  alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  ALPHA_INT columns = args_get_columns(argc, argv, k);
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);
  ALPHA_INT rowsx = k, rowsy = m;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rowsx = m;
    rowsy = k;
  }
  ALPHA_INT ldx = columns, ldy = columns;
  if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = rowsx;
    ldy = rowsy;
  }
  size_t size_x = rowsx * columns;
  size_t size_y = rowsy * columns;
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);
  alpha_fill_random_d(x, 1, size_x);
  alpha_fill_random_d(y, 1, size_y);
  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, sky;
  alpha_call_exit(alphasparse_d_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_d_create_coo");
  alpha_call_exit(alphasparse_convert_sky(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                       descr.mode, &sky),
                "alphasparse_convert_csr");
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_call_exit(alphasparse_d_mm_plain(transA, alpha, sky, descr, layout, x,
                                      columns, ldx, beta, y, ldy),
                "alphasparse_d_mm");
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_d_mm");
  alphasparse_destroy(coo);
  alphasparse_destroy(sky);
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
  const double alpha = 3.;
  const double beta = 2.;
  printf("thread_num : %d\n", thread_num);
  double *alpha_y, *mkl_y;
  size_t size_alpha_y, size_mkl_y;
  alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);
  int status = 0;
  if (check) {
    mkl_mm(argc, argv, file, thread_num, alpha, beta, &mkl_y, &size_mkl_y);
    check_d(mkl_y, size_mkl_y, alpha_y, size_alpha_y);
    alpha_free(mkl_y);
  }
  alpha_free(alpha_y);
  return status;
}