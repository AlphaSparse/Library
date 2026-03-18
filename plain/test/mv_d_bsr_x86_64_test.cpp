#include "test_common.h"
/**
 * @brief ict mv csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static int iter;

static void mkl_mv_coo(const int argc, const char *argv[], const char *file, int thread_num,
                       const double alpha, const double beta, double **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  double *values;
  mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  size_t size_x = k;
  size_t size_y = m;
  if (transA == SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }

  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz);
  alpha_fill_random_d((double *)x, 1, size_x);
  alpha_fill_random_d((double *)y, 1, size_y);

  mkl_set_num_threads(thread_num);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, bsrA;
  mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
  mkl_sparse_convert_bsr(cooA, 2, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_d_mv(transA, alpha, cooA, descr, x, beta, y), "mkl_sparse_d_mv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_d_mv", total_time / iter);

  // alpha_timing_end(&timer);

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(bsrA);

  *ret_y = y;
  *ret_size_y = size_y;
  // for (ALPHA_INT k = 0; k < 20000; k++){
  //	printf("y[%d]=%f\n",k,y[k]);
  //}
  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
static void mkl_mv(const int argc, const char *argv[], const char *file, int thread_num,
                   const double alpha, const double beta, double **ret_y, size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  double *values;
  mkl_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  size_t size_x = k;
  size_t size_y = m;
  if (transA == SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }

  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz);
  alpha_fill_random_d((double *)x, 1, size_x);
  alpha_fill_random_d((double *)y, 1, size_y);

  mkl_set_num_threads(thread_num);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, bsrA;
  mkl_sparse_d_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
  mkl_sparse_convert_bsr(cooA, 4, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_d_mv(transA, alpha, bsrA, descr, x, beta, y), "mkl_sparse_d_mv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_d_mv", total_time / iter);

  // alpha_timing_end(&timer);

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(bsrA);

  *ret_y = y;
  *ret_size_y = size_y;
  // for (ALPHA_INT k = 0; k < 20000; k++){
  //	printf("y[%d]=%f\n",k,y[k]);
  //}
  alpha_free(x);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
static void alpha_mv(const int argc, const char *argv[], const char *file, int thread_num,
                   const double alpha, const double beta, double **ret_y, size_t *ret_size_y) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  double *values;
  alpha_read_coo_d(file, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_x = k;
  size_t size_y = m;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }
  double *x = (double*)alpha_memalign(sizeof(double) * size_x, DEFAULT_ALIGNMENT);
  double *y = (double*)alpha_memalign(sizeof(double) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d(values, 1, nnz);
  alpha_fill_random_d(x, 1, size_x);
  alpha_fill_random_d(y, 1, size_y);

  alpha_set_thread_num(thread_num);

  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, bsrA;
  alpha_call_exit(alphasparse_d_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_d_create_coo");
  alpha_call_exit(alphasparse_convert_bsr(cooA, 4, ALPHA_SPARSE_LAYOUT_ROW_MAJOR,
                                       ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
                "alphasparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_d_mv_plain(transA, alpha, bsrA, descr, x, beta, y), "alphasparse_d_mv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "alphasparse_d_mv_plain", total_time / iter);
  ;

  // alpha_timing_end(&timer);
  alphasparse_destroy(cooA);
  alphasparse_destroy(bsrA);

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

  const double alpha = 3.;
  const double beta = 2.;

  const double mkl_alpha = 3.;
  const double mkl_beta = 2.;

  double *alpha_y;
  double *mkl_y;
  double *mkl_y20;
  double *mkl_ycoo;

  size_t size_alpha_y, size_mkl_y, size_mkl_y20, size_mkl_ycoo;

  printf("thread_num : %d\n", thread_num);

  int status = 0;

  alpha_mv(argc, argv, file, 1, alpha, beta, &alpha_y, &size_alpha_y);
  if (check) {
    mkl_mv(argc, argv, file, thread_num, mkl_alpha, mkl_beta, &mkl_y, &size_mkl_y);
    // //mkl multi-thread: for sanity check

    // mkl_mv(argc, argv, file, 20, mkl_alpha, mkl_beta, &mkl_y20,
    // &size_mkl_y20);

    // mkl_mv_coo(argc, argv, file, 1, mkl_alpha, mkl_beta, &mkl_ycoo,
    // &size_mkl_ycoo);

    // printf("mkl1 vs ict\n");
    status = check_d((double *)mkl_y, size_mkl_y, (double *)alpha_y, size_alpha_y);

    // printf("mkl20 vs ict\n");
    // status = check_d((double *)mkl_y20, size_mkl_y20, (double *)alpha_y,
    // size_alpha_y);

    // //sanity check
    // printf("mkl vs mkl_coo\n");
    // status = check_d((double *)mkl_y, size_mkl_y, (double *)mkl_ycoo,
    // size_mkl_ycoo); printf("mkl20 vs mkl_coo\n"); status = check_d((double
    // *)mkl_y20, size_mkl_y20, (double *)mkl_ycoo, size_mkl_ycoo);
    // // printf("mkl sanity check\n");
    // printf("mkl vs mkl20\n");
    // status = check_d((double *)mkl_y, size_mkl_y, (double *)mkl_y20,
    // size_mkl_y20);
    //  for(int i = 0 ; i < size_mkl_y ; i++){
    //     printf("mkl20 %lf ,ict %lf, diff %lf \n", ((double
    //     *)mkl_y20)[i],((double *)mkl_y)[i],((double *)mkl_y20)[i]-((double
    //     *)mkl_y)[i]);
    // }
    // printf("ict vs mkl_coo\n");
    // status = check_d((double *)alpha_y, size_alpha_y, (double *)mkl_ycoo,
    // size_mkl_ycoo);

    // for(int i = 0 ; i < size_mkl_y ; i++){
    //     printf("mkl %lf ,ict %lf, diff %lf\n", ((double *)mkl_y)[i],((double
    //     *)alpha_y)[i],((double *)mkl_y)[i] - ((double *)alpha_y)[i]);
    // }

    // for(int i = 0 ; i < size_mkl_y ; i++){
    //     printf("mkl20 %lf ,ict %lf, diff %lf \n", ((double
    //     *)mkl_y20)[i],((double *)alpha_y)[i],((double *)mkl_y20)[i]-((double
    //     *)alpha_y)[i]);
    // }

    alpha_free(mkl_y);
  }

  alpha_free(alpha_y);
  return status;
}