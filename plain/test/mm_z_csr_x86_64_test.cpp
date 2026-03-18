#include "test_common.h"
/**
 * @brief ict mm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, MKL_Complex16 alpha, MKL_Complex16 beta,
                   MKL_Complex16 **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex16 *values;
  mkl_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

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
  MKL_Complex16 *x =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex16 *y =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_z((ALPHA_Complex16 *)x, 1, size_x);
  alpha_fill_random_z((ALPHA_Complex16 *)y, 1, size_y);
  mkl_set_num_threads(thread_num);
  sparse_matrix_t coo, csr;
  mkl_call_exit(mkl_sparse_z_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                                        row_index, col_index, values),
                "mkl_sparse_z_create_coo");
  mkl_call_exit(
      mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr),
      "mkl_sparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_call_exit(mkl_sparse_z_mm(transA, alpha, csr, descr, layout, x, columns,
                                ldx, beta, y, ldy),
                "mkl_sparse_z_mm");

  alpha_timing_end(&timer);
  printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
         alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

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
  ALPHA_Complex16 *x =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_x, DEFAULT_ALIGNMENT);
  ALPHA_Complex16 *y =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_z(x, 1, size_x);
  alpha_fill_random_z(y, 1, size_y);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, csr;
  alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(
      alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csr),
      "alphasparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_z_mm_plain(transA, alpha, csr, descr, layout, x,
                                      columns, ldx, beta, y, ldy),
                "alphasparse_z_mm");

  alpha_timing_end(&timer);
  printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
         alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));

  alphasparse_destroy(coo);
  alphasparse_destroy(csr);

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

  printf("%d,", thread_num);

  ALPHA_Complex16 *alpha_y, *alpha_x;
  MKL_Complex16 *mkl_y;
  size_t size_alpha_x;
  ALPHA_INT ldx, ldy;
  size_t size_alpha_y, size_mkl_y;

  alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_x, &size_alpha_x, &ldx,
         &alpha_y, &size_alpha_y, &ldy);

  int status = 0;
  if (check) {
    printf(",");
    mkl_mm(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y,
           &size_mkl_y);
    printf(",");
    //        check_d(double *)mkl_y, size_mkl_y * 2, (double *)alpha_y,
    //        size_alpha_y * 2);
    status = check_z_l3((ALPHA_Complex16 *)mkl_y, ldy, size_mkl_y, alpha_y, ldy,
                        size_alpha_y, NULL, alpha_x, ldx, alpha_y, ldy, alpha, beta,
                        argc, argv);
    alpha_free(mkl_y);
  }
  printf("\n");
  alpha_free(alpha_y);
  return status;
}
