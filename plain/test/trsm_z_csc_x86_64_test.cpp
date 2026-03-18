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
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  MKL_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  MKL_Complex16 *x = (MKL_Complex16 *)alpha_malloc(size_x * sizeof(MKL_Complex16));
  MKL_Complex16 *y = (MKL_Complex16 *)alpha_malloc(size_y * sizeof(MKL_Complex16));
  alpha_fill_random_d((double *)x, 1, size_x * 2);

  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);
  sparse_index_base_t base_mkl = SPARSE_INDEX_BASE_ZERO;

  int ldx = columns, ldy = columns;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
    base_mkl = SPARSE_INDEX_BASE_ONE;
  }

  mkl_set_num_threads(thread_num);
  alphasparse_matrix_t cooA, cscA;
  alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA),
      "alphasparse_convert_csc");
  sparse_matrix_t cscB;
  mkl_sparse_z_create_csc(
      &cscB, base_mkl, m, k, ((spmat_csc_z_t *)cscA->mat)->cols_start,
      ((spmat_csc_z_t *)cscA->mat)->cols_end,
      ((spmat_csc_z_t *)cscA->mat)->row_indx,
      (MKL_Complex16 *)((spmat_csc_z_t *)cscA->mat)->values);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_z_trsm(transA, alpha, cscB, descr, layout, x,
                                  columns, ldx, y, ldy),
                "mkl_sparse_z_trsm");
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_z_trsm");

  alphasparse_destroy(cooA);
  alphasparse_destroy(cscA);
  mkl_sparse_destroy(cscB);

  *ret = y;
  *size = size_y;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

// static void alpha_trsm(const int argc, const char *argv[], const char *file,
// int thread_num, const ALPHA_Complex16 alpha, ALPHA_Complex16 **ret, size_t *size)
static void alpha_trsm(const int argc, const char *argv[], const char *file,
                     int thread_num, ALPHA_Complex16 alpha, ALPHA_Complex16 **ret_x,
                     size_t *ret_size_x, ALPHA_INT *ret_ldx,
                     ALPHA_Complex16 **ret_y, size_t *ret_size_y,
                     ALPHA_INT *ret_ldy) {
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

  alphasparse_matrix_t cooA, cscA;
  alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA),
      "alphasparse_convert_csc");
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_call_exit(alphasparse_z_trsm_plain(transA, alpha, cscA, descr, layout, x,
                                        columns, ldx, y, ldy),
                "alphasparse_z_trsm_plain");
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_z_trsm_plain");
  alphasparse_destroy(cooA);
  alphasparse_destroy(cscA);

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

  const ALPHA_Complex16 alpha_alpha = {.real = 2, .imag = 2};
  const MKL_Complex16 mkl_alpha = {.real = 2, .imag = 2};

  printf("thread_num : %d\n", thread_num);

  ALPHA_Complex16 *alpha_y;
  MKL_Complex16 *mkl_y;
  size_t size_alpha_y, size_mkl_y;
  ALPHA_Complex16 *alpha_x;
  size_t size_alpha_x;
  ALPHA_INT ldx, ldy;
  ALPHA_Complex16 zero = {0, 0};

  int status = 0;
  // alpha_trsm(argc, argv, file, thread_num, alpha_alpha, &alpha_y, &size_alpha_y);
  // alpha_trsm(argc, argv, file, thread_num, alpha_alpha, &alpha_x, &size_alpha_x,
  // &ldx, &alpha_y, &size_alpha_y, &ldy);

  if (check) {
    mkl_trsm(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
    // status = check_d((double *)mkl_y, size_mkl_y * 2, (double *)alpha_y,
    // size_alpha_y * 2); status = check_z_l3((ALPHA_Complex16 *)mkl_y, ldy,
    // size_mkl_y, alpha_y, ldy, size_alpha_y, NULL, alpha_x, ldx, alpha_y, ldy,
    // alpha_alpha, zero, argc, argv);
    alpha_free(mkl_y);
  }

  // alpha_free(alpha_y);
  return status;
}
