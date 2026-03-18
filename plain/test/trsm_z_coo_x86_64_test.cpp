#include "test_common.h"
/**
 * @brief ict trsm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_trsm(const int argc, const char *argv[], const char *file,
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
  MKL_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  MKL_Complex16 *x = (MKL_Complex16 *)alpha_malloc(size_x * sizeof(MKL_Complex16));
  MKL_Complex16 *y = (MKL_Complex16 *)alpha_malloc(size_y * sizeof(MKL_Complex16));
  alpha_fill_random_d((double *)x, 1, 2 * size_x);

  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  int ldx = columns, ldy = columns;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
  }

  mkl_set_num_threads(thread_num);
  sparse_matrix_t cooA, csrA;
  mkl_sparse_z_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_z_trsm(transA, alpha, csrA, descr, layout, x,
                                  columns, ldx, y, ldy),
                "mkl_sparse_z_trsm");
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

static void alpha_trsm(const int argc, const char *argv[], const char *file,
                     int thread_num, const ALPHA_Complex16 alpha,
                     ALPHA_Complex16 **ret, size_t *size) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  ALPHA_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  ALPHA_Complex16 *x = (ALPHA_Complex16 *)alpha_malloc(size_x * sizeof(ALPHA_Complex16));
  ALPHA_Complex16 *y = (ALPHA_Complex16 *)alpha_malloc(size_y * sizeof(ALPHA_Complex16));
  alpha_fill_random_z(x, 1, size_x);

  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  int ldx = columns, ldy = columns;
  if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
  }
  alpha_set_thread_num(thread_num);

  alphasparse_matrix_t cooA;
  alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_z_create_coo");
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_call_exit(alphasparse_z_trsm_plain(transA, alpha, cooA, descr, layout, x,
                                        columns, ldx, y, ldy),
                "alphasparse_z_trsm_plain");
  alpha_timing_end(&timer);
  printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
         alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));
  alphasparse_destroy(cooA);

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
  bool check = args_get_if_check(argc, argv);

  const ALPHA_Complex16 alpha_alpha = {.real = 2, .imag = 2};
  const MKL_Complex16 mkl_alpha = {.real = 2, .imag = 2};

  printf("%d,\n", thread_num);

  ALPHA_Complex16 *alpha_y;
  MKL_Complex16 *mkl_y;
  size_t size_alpha_y, size_mkl_y;

  int status = 0;
  alpha_trsm(argc, argv, file, thread_num, alpha_alpha, &alpha_y, &size_alpha_y);

  if (check) {
    alpha_clear_cache();
    printf(",");
    mkl_trsm(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
    printf(",");

    status = check_d((double *)mkl_y, 2 * size_mkl_y, (double *)alpha_y,
                     2 * size_alpha_y);
    alpha_free(mkl_y);
  }

  alpha_free(alpha_y);
  printf("\n");
  return status;
}