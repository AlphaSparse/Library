#include <alphasparse.h>
#include <stdio.h>

#include "../include/test_common.h"

#ifdef __MKL__
static void mkl_trsv(const int argc, const char *argv[], const char *file, int thread_num,
                     const float alpha, float **ret, size_t *size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  float *values;
  mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
  if (m != k) {
    printf("sparse matrix must be Square matrix but (%d,%d)\n", (int)m, (int)k);
    exit(-1);
  }
  for (MKL_INT i = 0; i < nnz; i++) {
    if (row_index[i] == col_index[i]) {
      values[i] += 1.0;
    }
  }
  size_t size_x = k;
  size_t size_y = m;
  float *x = (float *)alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
  float *y = (float *)alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);
  alpha_fill_random_s((float *)x, 1, size_x);

  mkl_set_num_threads(thread_num);

  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  struct matrix_descr descr = mkl_args_get_matrix_descrA(argc, argv);

  sparse_matrix_t cooA, csrA;
  mkl_sparse_s_create_coo(&cooA, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index, col_index, values);
  mkl_sparse_convert_csr(cooA, SPARSE_OPERATION_NON_TRANSPOSE, &csrA);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_call_exit(mkl_sparse_s_trsv(transA, alpha, csrA, descr, x, y), "mkl_sparse_s_trsv");
  alpha_timing_end(&timer);

  printf("%s time : %lf[ms]\n", "mkl_sparse_sv", (alpha_timing_elapsed_time(&timer)) * 1000);

  mkl_sparse_destroy(cooA);
  mkl_sparse_destroy(csrA);

  *ret = y;
  *size = size_y;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
#endif

static void alpha_trsv(const int argc, const char *argv[], const char *file, int thread_num,
                     const float alpha, float **ret, size_t *size, float **ret_x,
                     size_t *ret_size_x) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  float *values;
  alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
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
  float *x = (float *)alpha_memalign(sizeof(float) * size_x, DEFAULT_ALIGNMENT);
  float *y = (float *)alpha_memalign(sizeof(float) * size_y, DEFAULT_ALIGNMENT);
  for (ALPHA_INT i = 0; i < nnz; i++) {
    if (row_index[i] == col_index[i]) {
      values[i] += 1.0;
    }
  }
  alpha_fill_random_s(x, 1, size_x);

  alpha_set_thread_num(thread_num);
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  struct alpha_matrix_descr descr = alpha_args_get_matrix_descrA(argc, argv);

  alphasparse_matrix_t cooA, csrA;
  alpha_call_exit(alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_s_create_coo");
  alpha_call_exit(alphasparse_convert_csr(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
                "alphasparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_s_trsv(transA, alpha, csrA, descr, x, y), "alphasparse_s_trsv");

  alpha_timing_end(&timer);
  // printf("%lf,%lf", alpha_timing_elapsed_time(&timer),
  //        alpha_timing_gflops(&timer, (double)nnz * k * 2 + m * k));
  printf("%s time : %lf[ms]\n", "alphasparse_sv", (alpha_timing_elapsed_time(&timer)) * 1000);

  alphasparse_destroy(cooA);
  alphasparse_destroy(csrA);

  *ret = y;
  *size = size_y;
  *ret_x = x;
  *ret_size_x = size_x;

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

  const float alpha_alpha = 2.f;
  const float mkl_alpha = 2.f;

  printf("thread %d\n", thread_num);
  float alpha = 0.f;
  float beta = 0.f;
  float *alpha_y, *alpha_x;
  float *mkl_y;
  size_t size_alpha_y, size_mkl_y;
  size_t size_alpha_x;

  alpha_trsv(argc, argv, file, thread_num, alpha_alpha, &alpha_y, &size_alpha_y, &alpha_x, &size_alpha_x);
  int status = 0;
  if (check) {
    alpha_clear_cache();
#ifdef __MKL__
    mkl_trsv(argc, argv, file, thread_num, mkl_alpha, &mkl_y, &size_mkl_y);
#endif
    status = check_s((float *)mkl_y, size_mkl_y, (float *)alpha_y, size_alpha_y);
    alpha_free(mkl_y);
  }
  printf("\n");
  alpha_free(alpha_y);

  return status;
}