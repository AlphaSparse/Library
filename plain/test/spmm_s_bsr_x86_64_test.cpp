#include "test_common.h"
/**
 * @brief ict spmm csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static int iter;
#define BLOCK_SIZE 4
#define ROW_MAJOR 0
#define COL_MAJOR 1

static void mkl_spmm(const int argc, const char *argv[], const char *file,
                     int thread_num, sparse_index_base_t *ret_index,
                     MKL_INT *ret_rows, MKL_INT *ret_cols,
                     MKL_INT **ret_rows_start, MKL_INT **ret_rows_end,
                     MKL_INT **ret_col_index, float **ret_values) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_INT rowsA, rowsB, colsA, colsB;
  float *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = NULL;
  sparse_matrix_t cooA, bsrA, cooB, bsrB, result;
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = SPARSE_LAYOUT_ROW_MAJOR;

  mkl_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_fill_random_s((float *)values, 1, nnz);
  mkl_set_num_threads(thread_num);
  if (transA == SPARSE_OPERATION_TRANSPOSE) {
    rowsA = k;
    colsA = m;
    fileB = args_get_data_fileA(argc, argv);
  } else {
    rowsA = m;
    colsA = k;
    fileB = args_get_data_fileB(argc, argv);
  }

  mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_call_exit(
      mkl_sparse_convert_bsr(cooA, BLOCK_SIZE, SPARSE_LAYOUT_ROW_MAJOR,
                             SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
      "mkl_sparse_convert_bsr");
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);

  mkl_read_coo(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_fill_random_s((float *)values, 1, nnz);
  rowsB = m;
  colsB = k;
  printf("rowsA %d, colsA %d, rowsB %d, colsB %d\n", rowsA, colsA, rowsB,
         colsB);

  mkl_sparse_s_create_coo(&cooB, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_call_exit(
      mkl_sparse_convert_bsr(cooB, BLOCK_SIZE, SPARSE_LAYOUT_ROW_MAJOR,
                             SPARSE_OPERATION_NON_TRANSPOSE, &bsrB),
      "mkl_sparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_sparse_spmm(transA, bsrA, bsrB, &result);
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_spmm",
         total_time / iter);
  // alpha_timing_end(&timer);

  mkl_sparse_order(result);

  sparse_layout_t mkl_block_layout;
  MKL_INT mkl_block_size;
  mkl_call_exit(
      mkl_sparse_s_export_bsr(result, ret_index, &mkl_block_layout, ret_rows,
                              ret_cols, &mkl_block_size, ret_rows_start,
                              ret_rows_end, ret_col_index, ret_values),
      "mkl_sparse_s_export_bsr");

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(cooB);

  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

static void alpha_spmm(const int argc, const char *argv[], const char *file,
                     int thread_num, alphasparseIndexBase_t *ret_index,
                     ALPHA_INT *ret_rows, ALPHA_INT *ret_cols,
                     ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end,
                     ALPHA_INT **ret_col_index, float **ret_values, int flag) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_INT rowsA, rowsB, colsA, colsB;
  float *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = NULL;
  alphasparse_matrix_t cooA, bsrA, cooB, bsrB, result;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  alphasparse_layout_t layout = ALPHA_SPARSE_LAYOUT_ROW_MAJOR;

  alpha_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_fill_random_s((float *)values, 1, nnz);
  // alpha_set_num_threads(thread_num);
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    rowsA = k;
    colsA = m;
    fileB = args_get_data_fileA(argc, argv);
  } else {
    rowsA = m;
    colsA = k;
    fileB = args_get_data_fileB(argc, argv);
  }

  alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                          row_index, col_index, values);
  alpha_call_exit(
      alphasparse_convert_bsr(cooA, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
      "alphasparse_convert_bsr");
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);

  alpha_read_coo(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_fill_random_s((float *)values, 1, nnz);
  rowsB = m;
  colsB = k;
  printf("rowsA %d, colsA %d, rowsB %d, colsB %d\n", rowsA, colsA, rowsB,
         colsB);

  alphasparse_s_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                          row_index, col_index, values);
  alpha_call_exit(
      alphasparse_convert_bsr(cooB, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrB),
      "alphasparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alphasparse_spmm_plain(transA, bsrA, bsrB, &result);
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "alphasparse_spmm_plain",
         total_time / iter);
  ;
  // alpha_timing_end(&timer);

  // alphasparse_order(result);

  alphasparse_layout_t alpha_block_layout;
  ALPHA_INT alpha_block_size;
  alpha_call_exit(
      alphasparse_s_export_bsr(result, ret_index, &alpha_block_layout, ret_rows,
                              ret_cols, &alpha_block_size, ret_rows_start,
                              ret_rows_end, ret_col_index, ret_values),
      "alphasparse_s_export_bsr");

  alphasparse_destroy(cooA);
  alphasparse_destroy(cooB);

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

  // return row major
  sparse_index_base_t mkl_index;
  MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
  float *mkl_values;

  alphasparseIndexBase_t alpha_index;
  ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
  float *alpha_values;

  /*
  row major
  */

  int status = 0;
  if (check) {
    mkl_spmm(argc, argv, file, thread_num, &mkl_index, &mkl_rows, &mkl_cols,
             &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values);
    // alpha_spmm(argc, argv, file, thread_num, &alpha_index, &alpha_rows, &alpha_cols,
    // &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, ROW_MAJOR);
    // int mkl_nnz = mkl_rows_end[mkl_rows - 1];
    // int alpha_nnz = alpha_rows_end[alpha_rows/BLOCK_SIZE - 1];
    // status = check_s((float *)mkl_values, mkl_nnz*BLOCK_SIZE*BLOCK_SIZE,
    // (float *)alpha_values, alpha_nnz*BLOCK_SIZE*BLOCK_SIZE);
  }

  return status;
}
