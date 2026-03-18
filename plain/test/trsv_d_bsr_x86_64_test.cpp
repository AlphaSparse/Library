#include "test_common.h"
/**
 * @brief ict trsv csr test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static int iter;
#include <math.h>
#define BLOCK_SIZE 4
#define ROW_MAJOR false
#define COL_MAJOR true

static void mkl_trsv(const int argc, const char *argv[], const char *file,
                     int thread_num, const double alpha, double **ret,
                     size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  double *values;
  mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  // for(MKL_INT i = 0; i < nnz; i++)
  // {
  //     if(row_index[i] == col_index[i])
  //     {
  //         values[i] += 1.0;
  //     }
  // }
  size_t size_x = k;
  size_t size_y = m;
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);
  alpha_fill_random_d((double *)x, 1, size_x);

  mkl_set_num_threads(thread_num);

  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, csrA, bsrA;
  mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                          col_index, values);
  // mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  mkl_call_exit(
      mkl_sparse_convert_bsr(cooA, BLOCK_SIZE, SPARSE_LAYOUT_ROW_MAJOR,
                             SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
      "mkl_sparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);
  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_d_trsv(transA, alpha, bsrA, descr, x, y),
                  "mkl_sparse_d_trsv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_d_trsv",
         total_time / iter);
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

static void alpha_trsv(const int argc, const char *argv[], const char *file,
                     int thread_num, const double alpha, double **ret,
                     size_t *size, bool flag) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  double *values;
  alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
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
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d(x, 1, size_x);

  alphasparse_layout_t layout;
  if (flag == ROW_MAJOR)
    layout = ALPHA_SPARSE_LAYOUT_ROW_MAJOR;
  else
    layout = ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR;
  alpha_set_thread_num(thread_num);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, bsrA;
  alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_d_create_coo");
  alpha_call_exit(
      alphasparse_convert_bsr(cooA, BLOCK_SIZE, layout,
                             ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
      "alphasparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_d_trsv_plain(transA, alpha, bsrA, descr, x, y),
                  "alphasparse_d_trsv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter,
         "alphasparse_d_trsv_plain", total_time / iter);
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

  const double alpha_alpha = 3.;
  const double mkl_alpha = 3.;

  printf("thread_num : %d\n", thread_num);

  // block row major
  printf("block row major begin\n");
  double *alpha_y;
  double *mkl_y;
  size_t size_alpha_y, size_mkl_y;

  int status = 0;
  if (check) {
    mkl_trsv(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
    alpha_trsv(argc, argv, file, thread_num, alpha_alpha, &alpha_y, &size_alpha_y,
             ROW_MAJOR);

    status = check_d((double *)mkl_y, size_mkl_y, (double *)alpha_y, size_alpha_y);
    // double mkl_nor2 = 0, alpha_nor2 = 0;
    // double * fmkl_y = (double *)mkl_y, * falpha_y = (double *)alpha_y;
    // for(size_t i = 0; i < size_mkl_y; i++)
    // {
    //     mkl_nor2 += fmkl_y[i] * fmkl_y[i];
    //     alpha_nor2 += falpha_y[i] * falpha_y[i];
    // }
    // printf("mkl norm 2 = %f, ict norm2 = %f \n",sqrt(mkl_nor2),
    // sqrt(alpha_nor2));
    alpha_free(mkl_y);
  }
  printf("block row major end\n");

  // block column major
  // printf("block column major begin\n");
  // double *alpha_y_col;
  // double *mkl_y_col;
  // size_t size_alpha_y_col, size_mkl_y_col;

  // if (check)
  // {
  //     mkl_trsv(argc, argv, file, thread_num, mkl_alpha, &mkl_y_col,
  //     &size_mkl_y_col); alpha_trsv(argc, argv, file, thread_num, alpha_alpha,
  //     &alpha_y_col, &size_alpha_y_col, COL_MAJOR); status = check_d((double
  //     *)mkl_y_col, size_mkl_y_col, (double *)alpha_y_col, size_alpha_y_col);
  //     //for(int i=0; i<size_mkl_y_col; i++)
  //     {
  //         //printf("mkl:%f, ict:%f\n", mkl_y_col[size_mkl_y_col-2],
  //         alpha_y_col[size_mkl_y_col-2]);
  //         //printf("mkl:%f, ict:%f\n", mkl_y_col[size_mkl_y_col-1],
  //         alpha_y_col[size_mkl_y_col-1]);
  //     }

  //     alpha_free(mkl_y_col);
  // }
  // printf("block column major end\n\n\n");

  // alpha_free(alpha_y);
  // alpha_free(alpha_y_col);

  return status;
}
