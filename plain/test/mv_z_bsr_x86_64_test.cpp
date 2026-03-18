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
                       const MKL_Complex16 alpha, const MKL_Complex16 beta, MKL_Complex16 **ret_y,
                       size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex16 *values;
  mkl_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  size_t size_x = k;
  size_t size_y = m;
  if (transA == SPARSE_OPERATION_TRANSPOSE || transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }

  MKL_Complex16 *x = (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex16 *y = (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz * 2);
  alpha_fill_random_d((double *)x, 1, size_x * 2);
  alpha_fill_random_d((double *)y, 1, size_y * 2);

  mkl_set_num_threads(thread_num);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, bsrA;
  mkl_sparse_z_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
  mkl_sparse_convert_bsr(cooA, 2, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_z_mv(transA, alpha, cooA, descr, x, beta, y), "mkl_sparse_z_mv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_z_mv", total_time / iter);

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
                   const MKL_Complex16 alpha, const MKL_Complex16 beta, MKL_Complex16 **ret_y,
                   size_t *ret_size_y) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex16 *values;
  mkl_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  size_t size_x = k;
  size_t size_y = m;
  if (transA == SPARSE_OPERATION_TRANSPOSE || transA == SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }

  MKL_Complex16 *x = (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_x, DEFAULT_ALIGNMENT);
  MKL_Complex16 *y = (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_d((double *)values, 1, nnz * 2);
  alpha_fill_random_d((double *)x, 1, size_x * 2);
  alpha_fill_random_d((double *)y, 1, size_y * 2);

  mkl_set_num_threads(thread_num);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, bsrA;
  mkl_sparse_z_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
  mkl_sparse_convert_bsr(cooA, 4, SPARSE_LAYOUT_ROW_MAJOR, SPARSE_OPERATION_NON_TRANSPOSE, &bsrA);

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    mkl_call_exit(mkl_sparse_z_mv(transA, alpha, bsrA, descr, x, beta, y), "mkl_sparse_z_mv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "mkl_sparse_z_mv", total_time / iter);

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
                   const ALPHA_Complex16 alpha, const ALPHA_Complex16 beta, ALPHA_Complex16 **ret_y,
                   size_t *ret_size_y) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex16 *values;
  alpha_read_coo_z(file, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_x = k;
  size_t size_y = m;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  if (transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    size_x = m;
    size_y = k;
  }
  ALPHA_Complex16 *x = (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_x, DEFAULT_ALIGNMENT);
  ALPHA_Complex16 *y = (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * size_y, DEFAULT_ALIGNMENT);

  alpha_fill_random_z(values, 1, nnz);
  alpha_fill_random_z(x, 1, size_x);
  alpha_fill_random_z(y, 1, size_y);

  alpha_set_thread_num(thread_num);

  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, bsrA;
  alpha_call_exit(alphasparse_z_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_z_create_coo");
  alpha_call_exit(alphasparse_convert_bsr(cooA, 4, ALPHA_SPARSE_LAYOUT_ROW_MAJOR,
                                       ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &bsrA),
                "alphasparse_convert_bsr");

  alpha_timer_t timer;
  // alpha_timing_start(&timer);

  double total_time = 0.;
  for (int i = 0; i < iter; i++) {
    alpha_clear_cache();
    alpha_timing_start(&timer);
    alpha_call_exit(alphasparse_z_mv_plain(transA, alpha, bsrA, descr, x, beta, y), "alphasparse_z_mv");
    alpha_timing_end(&timer);
    total_time += alpha_timing_elapsed_time(&timer);
  }
  printf("iter is %d, %s avg time : %lf[sec]\n", iter, "alphasparse_z_mv_plain", total_time / iter);
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

  const ALPHA_Complex16 alpha = {.real = 3., .imag = 3.};
  const ALPHA_Complex16 beta = {.real = 2, .imag = 0.};

  const MKL_Complex16 mkl_alpha = {.real = 3., .imag = 3.};
  const MKL_Complex16 mkl_beta = {.real = 2, .imag = 0};

  ALPHA_Complex16 *alpha_y;
  MKL_Complex16 *mkl_y;
  MKL_Complex16 *mkl_y20;
  MKL_Complex16 *mkl_ycoo;

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
    status = check_d((double *)mkl_y, size_mkl_y * 2, (double *)alpha_y, size_alpha_y * 2);

    // printf("mkl20 vs ict\n");
    // status = check_d((double *)mkl_y20, size_mkl_y20 * 2, (double *)alpha_y,
    // size_alpha_y * 2);

    // //sanity check
    // printf("mkl vs mkl_coo\n");
    // status = check_d((double *)mkl_y, size_mkl_y * 2, (double *)mkl_ycoo,
    // size_mkl_ycoo * 2); printf("mkl20 vs mkl_coo\n"); status =
    // check_d((double *)mkl_y20, size_mkl_y20 * 2, (double *)mkl_ycoo,
    // size_mkl_ycoo * 2);
    // // printf("mkl sanity check\n");
    // printf("mkl vs mkl20\n");
    // status = check_d((double *)mkl_y, size_mkl_y * 2, (double *)mkl_y20,
    // size_mkl_y20 * 2);
    //  for(int i = 0 ; i < size_mkl_y*2 ; i++){
    //     printf("mkl20 %lf ,ict %lf, diff %lf \n", ((double
    //     *)mkl_y20)[i],((double *)mkl_y)[i],((double *)mkl_y20)[i]-((double
    //     *)mkl_y)[i]);
    // }
    // printf("ict vs mkl_coo\n");
    // status = check_d((double *)alpha_y, size_alpha_y * 2, (double *)mkl_ycoo,
    // size_mkl_ycoo * 2);

    // for(int i = 0 ; i < size_mkl_y*2 ; i++){
    //     printf("mkl %lf ,ict %lf, diff %lf\n", ((double *)mkl_y)[i],((double
    //     *)alpha_y)[i],((double *)mkl_y)[i] - ((double *)alpha_y)[i]);
    // }

    // for(int i = 0 ; i < size_mkl_y*2 ; i++){
    //     printf("mkl20 %lf ,ict %lf, diff %lf \n", ((double
    //     *)mkl_y20)[i],((double *)alpha_y)[i],((double *)mkl_y20)[i]-((double
    //     *)alpha_y)[i]);
    // }

    alpha_free(mkl_y);
  }

  alpha_free(alpha_y);
  return status;
}