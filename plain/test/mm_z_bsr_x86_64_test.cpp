#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"

static int iter;

#define BLOCK_SIZE 4
#define BLOCK_LAYOUT SPARSE_LAYOUT_COLUMN_MAJOR
#define ALPHA_BLOCK_LAYOUT ALPHA_SPARSE_LAYOUT_ROW_MAJOR
#define MKL_BASE SPARSE_INDEX_BASE_ONE
#define ALPHA_BASE ALPHA_SPARSE_INDEX_BASE_ZERO

static void mkl_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, MKL_Complex16 alpha, MKL_Complex16 beta,
                   MKL_Complex16 **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex16 *values;
  sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO;
  mkl_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

  MKL_INT columns = args_get_columns(argc, argv, k);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  MKL_INT ldx = columns, ldy = columns;
  MKL_INT rows = m, cols = k;
  if (transA == SPARSE_OPERATION_TRANSPOSE ||
      transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rows = k;
    cols = m;
  }

  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = cols;
    ldy = rows;
    base = SPARSE_INDEX_BASE_ONE;
  }

  if (base == SPARSE_INDEX_BASE_ONE) {
    for (int i = 0; i < nnz; i++) {
      row_index[i]++;
      col_index[i]++;
    }
  }
  size_t size_x = cols * columns;
  size_t size_y = rows * columns;
  MKL_Complex16 *x =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex16 *y =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz * 2);
  alpha_fill_random_d((double *)x, 1, size_x * 2);
  alpha_fill_random_d((double *)y, 1, size_y * 2);

  mkl_set_num_threads(thread_num);
  sparse_matrix_t coo, bsr;
  mkl_call_exit(mkl_sparse_z_create_coo(&coo, base, m, k, nnz, row_index,
                                        col_index, values),
                "mkl_sparse_z_create_coo");
  mkl_call_exit(mkl_sparse_convert_bsr(coo, BLOCK_SIZE, layout,
                                       SPARSE_OPERATION_NON_TRANSPOSE, &bsr),
                "mkl_sparse_convert_bsr");
  // mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE,
  // &bsr), "mkl_sparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_z_mm(transA, alpha, bsr, descr, layout, x, columns,
                                  ldx, beta, y, ldy),
                  "mkl_sparse_z_mm");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_z_mm",
         total_time / iter);

  // alpha_timing_end(&timer);

  *ret_y = y;
  *ret_size_y = size_y;

  mkl_sparse_destroy(coo);
  mkl_sparse_destroy(bsr);
  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

static void alpha_mm(const int argc, const char *argv[], const char *file,
                   int thread_num, ALPHA_Complex16 alpha, ALPHA_Complex16 beta,
                   ALPHA_Complex16 **ret_y, size_t *ret_size_y) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

  ALPHA_INT columns = args_get_columns(argc, argv, k);
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  ALPHA_INT ldx = columns, ldy = columns;
  ALPHA_INT rows = m, cols = k;
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rows = k;
    cols = m;
  }

  if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = cols;
    ldy = rows;
  }

  size_t size_x = cols * columns;
  size_t size_y = rows * columns;
  ALPHA_Complex16 *x =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_x, DEFAULT_ALIGNMENT);
  ALPHA_Complex16 *y =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz * 2);
  alpha_fill_random_d((double *)x, 1, size_x * 2);
  alpha_fill_random_d((double *)y, 1, size_y * 2);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, bsr;
  alpha_call_exit(alphasparse_z_create_coo(&coo, ALPHA_BASE, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(
      alphasparse_convert_bsr(coo, BLOCK_SIZE, layout,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsr),
      "alphasparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  //    // printf("alphasparse_c_mm_plain called\n");

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_z_mm_plain(transA, alpha, bsr, descr, layout, x,
                                        columns, ldx, beta, y, ldy),
                  "alphasparse_z_mm");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "alphasparse_z_mm_plain",
         total_time / iter);
  ;

  // alpha_timing_end(&timer);

  alphasparse_destroy(coo);
  alphasparse_destroy(bsr);

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
  iter = args_get_iter(argc, argv);

  const ALPHA_Complex16 alpha = {.real = 3., .imag = 3.};
  const ALPHA_Complex16 beta = {.real = 2., .imag = 2.};
  const MKL_Complex16 mkl_alpha = {.real = 3., .imag = 3.};
  const MKL_Complex16 mkl_beta = {.real = 2., .imag = 2.};

  printf("thread_num : %d\n", thread_num);

  ALPHA_Complex16 *alpha_y;
  MKL_Complex16 *mkl_y;
  size_t size_alpha_y, size_mkl_y;

  int status = 0;
  if (check) {
    mkl_mm(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y,
           &size_mkl_y);
    // alpha_clear_cache();
    alpha_mm(argc, argv, file, thread_num, alpha, beta, &alpha_y, &size_alpha_y);
    check_d((double *)mkl_y, 2 * size_mkl_y, (double *)alpha_y, 2 * size_alpha_y);

    alpha_free(mkl_y);
    alpha_free(alpha_y);
  }

  return status;
}
