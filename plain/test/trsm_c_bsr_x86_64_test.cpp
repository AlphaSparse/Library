#include "test_common.h"
/**
 * @brief ict trsm bsr test
 * @author jaxxxxxo
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#define BLOCK_SIZE 4
static int iter;

static void mkl_trsm(const int argc, const char *argv[], const char *file, int thread_num,
                     const MKL_Complex8 alpha, MKL_Complex8 **ret, size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex8 *values;
  mkl_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  MKL_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  MKL_Complex8 *x = (MKL_Complex8 *)alpha_malloc(size_x * sizeof(MKL_Complex8));
  MKL_Complex8 *y = (MKL_Complex8 *)alpha_malloc(size_y * sizeof(MKL_Complex8));
  alpha_fill_random_s((float *)x, 1, size_x * 2);

  sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO;
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  int ldx = columns, ldy = columns;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
    base = SPARSE_INDEX_BASE_ONE;
  }

  if (base == SPARSE_INDEX_BASE_ONE) {
    for (int i = 0; i < nnz; i++) {
      row_index[i]++;
      col_index[i]++;
    }
  }
  mkl_set_num_threads(thread_num);
  sparse_matrix_t cooA, csrA, bsrA;
  mkl_sparse_c_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
  // mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  mkl_call_exit(
      mkl_sparse_convert_bsr(cooA, BLOCK_SIZE, layout, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
      "mkl_sparse_convert_bsr");
  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_c_trsm(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy),
                  "mkl_sparse_c_trsm");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_c_trsm", total_time / iter);
  // alpha_timing_end(&timer);
  mkl_sparse_destroy(cooA);
  // mkl_sparse_destroy(csrA);
  mkl_sparse_destroy(bsrA);

  *ret = y;
  *size = size_y;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

static void mkl_trsm_bsr(const int argc, const char *argv[], const char *file, int thread_num,
                         const MKL_Complex8 alpha, MKL_Complex8 **ret, size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex8 *values;
  mkl_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  MKL_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  MKL_Complex8 *x = (MKL_Complex8 *)alpha_malloc(size_x * sizeof(MKL_Complex8));
  MKL_Complex8 *y = (MKL_Complex8 *)alpha_malloc(size_y * sizeof(MKL_Complex8));
  alpha_fill_random_s((float *)x, 1, size_x * 2);

  // TODO block_layout == layout??
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  sparse_layout_t b_layout = layout;
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  // TODO mkl_args_get_blockSize()
  const MKL_INT block_size = 4;
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);
  sparse_index_base_t base = SPARSE_INDEX_BASE_ZERO;

  int ldx = columns, ldy = columns;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
    base = SPARSE_INDEX_BASE_ONE;
  }

  if (base == SPARSE_INDEX_BASE_ONE) {
    for (int i = 0; i < nnz; i++) {
      row_index[i]++;
      col_index[i]++;
    }
  }

  mkl_set_num_threads(thread_num);
  sparse_matrix_t cooA, bsrA;
  // #ifdef DEBUG
  //     printf(" mkl_sparse_c_create_coo starts \n");
  // #endif
  mkl_sparse_c_create_coo(&cooA, base, m, k, nnz, row_index, col_index, values);
  // #ifdef DEBUG
  //     printf(" mkl_sparse_c_create_coo ends \n");
  // #endif
  // #ifdef DEBUG
  //     printf(" mkl_sparse_convert_bsr starts \n");
  // #endif
  mkl_sparse_convert_bsr(cooA, block_size, b_layout, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);
  // #ifdef DEBUG
  //     printf(" mkl_sparse_convert_bsr ends \n");
  // #endif
  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  // #ifdef DEBUG
  //    //     printf(" mkl_sparse_c_trsm starts \n");
  // #endif

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_c_trsm(transA, alpha, bsrA, descr, layout, x, columns, ldx, y, ldy),
                  "mkl_sparse_c_trsm");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_c_trsm", total_time / iter);
  ldy = ldx;
  ldx = ldy;
  // #ifdef DEBUG
  //    //     printf(" mkl_sparse_c_trsm ends \n");
  // #endif
  // alpha_timing_end(&timer);
  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(bsrA);

  *ret = y;
  *size = size_y;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

static void alpha_trsm(const int argc, const char *argv[], const char *file, int thread_num,
                     const ALPHA_Complex8 alpha, ALPHA_Complex8 **ret, size_t *size) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex8 *values;
  alpha_read_coo_c(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  ALPHA_INT columns = args_get_columns(argc, argv, k);

  size_t size_x = k * columns;
  size_t size_y = m * columns;
  ALPHA_Complex8 *x = (ALPHA_Complex8 *)alpha_malloc(size_x * sizeof(ALPHA_Complex8));
  ALPHA_Complex8 *y = (ALPHA_Complex8 *)alpha_malloc(size_y * sizeof(ALPHA_Complex8));
  alpha_fill_random_c(x, 1, size_x);
  // #ifdef DEBUG
  //     printf("columns is %d alpha_trsm called x is:\n",columns);
  //     for(size_t i = 0 ; i < size_x;i++)
  //         printf("x[%ld],(%f,%f)\n",i, x[i].real, x[i].imag);
  // #endif
  // TODO block_layout == layout??
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  alphasparse_layout_t b_layout = layout;

  // TODO mkl_args_get_blockSize()
  const ALPHA_INT block_size = 4;
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  int ldx = columns, ldy = columns;
  if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = k;
    ldy = m;
  }
  alpha_set_thread_num(thread_num);

  alphasparse_matrix_t cooA, bsrA;
  alpha_call_exit(alphasparse_c_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_c_create_coo");
  alpha_call_exit(
      alphasparse_convert_bsr(cooA, block_size, b_layout, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
      "alphasparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alpha_call_exit(
        alphasparse_c_trsm_plain(transA, alpha, bsrA, descr, b_layout, x, columns, ldx, y, ldy),
        "alphasparse_c_trsm");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "alphasparse_c_trsm_plain",
         total_time / iter);
  ;
  // alpha_timing_end(&timer);

  alphasparse_destroy(cooA);
  alphasparse_destroy(bsrA);

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
  iter = args_get_iter(argc, argv);

  const ALPHA_Complex8 alpha_alpha = {.real = 2, .imag = 2};
  const MKL_Complex8 mkl_alpha = {.real = 2, .imag = 2};

  printf("thread_num : %d\n", thread_num);

  ALPHA_Complex8 *alpha_y;
  MKL_Complex8 *mkl_y;
  size_t size_alpha_y, size_mkl_y;

  int status = 0;

  alpha_trsm(argc, argv, file, thread_num, alpha_alpha, &alpha_y, &size_alpha_y);
  if (check) {
    mkl_trsm_bsr(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
    status = check_s((float *)mkl_y, size_mkl_y * 2, (float *)alpha_y, size_alpha_y * 2);

    alpha_free(mkl_y);
  }

  alpha_free(alpha_y);
  return status;
}