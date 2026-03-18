#include "test_common.h"
/**
 * @brief ict spmmd csr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static int iter;
#define BLOCK_SIZE 4
#define ROW_MAJOR 0
#define COL_MAJOR 1

static void mkl_spmmd(const int argc, const char *argv[], const char *file,
                      int thread_num, float **ret, size_t *ret_size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_INT rowsA, rowsB, colsA, colsB;
  float *values;

  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = NULL;
  sparse_matrix_t cooA, bsrA, cooB, bsrB;
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);

  mkl_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_fill_random_s((float *)values, 1, nnz);
  if (transA == SPARSE_OPERATION_TRANSPOSE) {
    rowsA = k;
    colsA = m;
    fileB = args_get_data_fileA(argc, argv);
  } else {
    rowsA = m;
    colsA = k;
    fileB = args_get_data_fileB(argc, argv);
  }

  mkl_set_num_threads(thread_num);

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

  mkl_sparse_s_create_coo(&cooB, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  mkl_call_exit(
      mkl_sparse_convert_bsr(cooB, BLOCK_SIZE, SPARSE_LAYOUT_ROW_MAJOR,
                             SPARSE_OPERATION_NON_TRANSPOSE, &bsrB),
      "mkl_sparse_convert_bsr");

  size_t size_C = rowsA * colsB;
  float *C = (float*)alpha_malloc(sizeof(float) * size_C);
  MKL_INT ldc;
  if (layout == SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldc = rowsA;
  } else {
    ldc = colsB;
  }
  printf("rowsA %d, colsA %d, rowsB %d, colsB %d ldc %d \n", rowsA, colsA,
         rowsB, colsB, ldc);

  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_s_spmmd(transA, bsrA, bsrB, layout, C, ldc),
                  "mkl_sparse_s_spmmd");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_s_spmmd",
         total_time / iter);
  // alpha_timing_end(&timer);
  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(bsrA);

  *ret = C;
  *ret_size = size_C;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

static void alpha_spmmd(const int argc, const char *argv[], const char *file,
                      int thread_num, float **ret, size_t *ret_size,
                      alphasparse_layout_t block_layout) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_INT rowsA, rowsB, colsA, colsB;
  float *values;

  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = NULL;
  alphasparse_matrix_t cooA, bsrA, cooB, bsrB;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

  alpha_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_fill_random_s((float *)values, 1, nnz);
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

  alphasparse_s_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                          row_index, col_index, values);
  alpha_call_exit(
      alphasparse_convert_bsr(cooB, BLOCK_SIZE, ALPHA_SPARSE_LAYOUT_ROW_MAJOR,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrB),
      "alphasparse_convert_bsr");

  size_t size_C = rowsA * colsB;
  float *C = (float*)alpha_malloc(sizeof(float) * size_C);
  ALPHA_INT ldc;
  if (layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldc = rowsA;
  } else {
    ldc = colsB;
  }
  printf("rowsA %d, colsA %d, rowsB %d, colsB %d ldc %d \n", rowsA, colsA,
         rowsB, colsB, ldc);

  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_s_spmmd_plain(transA, bsrA, bsrB, layout, C, ldc),
                  "alphasparse_s_spmmd");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter,
         "alphasparse_s_spmmd_plain", total_time / iter);
  ;
  // alpha_timing_end(&timer);
  alphasparse_destroy(cooA);
  alphasparse_destroy(bsrA);

  *ret = C;
  *ret_size = size_C;
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

  float *alpha_C;
  float *mkl_C;
  size_t size_mkl_C, size_alpha_C;
  printf("thread_num : %d\n", thread_num);

  int status = 0;
  {  // block row major

    if (check) {
      mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
      // alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C,
      // ALPHA_SPARSE_LAYOUT_ROW_MAJOR);

      // //for(int i = 0; i<100; i++)
      // {
      // //  printf("mkl:%f, ict:%f\n", (float)(mkl_C[i].real),
      // (float)(alpha_C[i].real));
      // }
      // status = check_s((float *)mkl_C, size_mkl_C, (float *)alpha_C,
      // size_alpha_C);
      alpha_free(mkl_C);
    }
  }

  // { // block col major
  //     alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C,
  //     ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR);

  //     if (check)
  //     {
  //         mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
  //         //for(int i = 0; i<100; i++)
  //     {
  //     //  printf("mkl:%f, ict:%f\n", (float)(mkl_C[i].real),
  //     (float)(alpha_C[i].real));
  //     }
  //     status = check_s((float *)mkl_C, size_mkl_C, (float *)alpha_C,
  //     size_alpha_C);
  //         alpha_free(mkl_C);
  //     }
  // }

  // alpha_free(alpha_C);
  return status;
}
