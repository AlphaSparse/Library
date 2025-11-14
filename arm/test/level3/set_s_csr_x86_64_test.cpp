#include "test_common.h"

#include <alphasparse.h>
#include <stdio.h>
#include <string.h>

#ifdef __MKL__
#include <mkl.h>
static void mkl_set(const int argc, const char *argv[], const char *file,
                    int thread_num, sparse_index_base_t *ret_index,
                    MKL_INT *ret_rows, MKL_INT *ret_cols,
                    MKL_INT **ret_rows_start, MKL_INT **ret_rows_end,
                    MKL_INT **ret_col_index, float **ret_values, MKL_INT off) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  float *values;
  mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

  mkl_set_num_threads(thread_num);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  sparse_matrix_t coo, csrA;
  mkl_call_exit(mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                                        row_index, col_index, values),
                "mkl_sparse_s_create_coo");
  mkl_call_exit(
      mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
      "mkl_sparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_call_exit(
      mkl_sparse_s_set_value(csrA, row_index[off], col_index[off], 12321.0f),
      "mkl_sparse_set");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_set");

  mkl_call_exit(mkl_sparse_s_export_csr(csrA, ret_index, ret_rows, ret_cols,
                                        ret_rows_start, ret_rows_end,
                                        ret_col_index, ret_values),
                "mkl_sparse_s_export_csr");

  float *y = (float *)alpha_memalign(sizeof(float) * nnz, DEFAULT_ALIGNMENT);
  memcpy(y, *ret_values, sizeof(float) * nnz);
  *ret_values = y;

  printf("MKL To be changed row %d col %d\n", row_index[off], col_index[off]);

  mkl_sparse_destroy(coo);
  mkl_sparse_destroy(csrA);

  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
#endif
static void alpha_set(const int argc, const char *argv[], const char *file,
                    int thread_num, alphasparseIndexBase_t *ret_index,
                    ALPHA_INT *ret_rows, ALPHA_INT *ret_cols,
                    ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end,
                    ALPHA_INT **ret_col_index, float **ret_values, ALPHA_INT off) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  float *values;
  alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, csrA;

  alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_s_create_coo");
  alpha_call_exit(
      alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
      "alphasparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_s_set_value(csrA, row_index[off],
                                             col_index[off], 12321.0f),
                "alphasparse_set");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_set_plain");

  alpha_call_exit(alphasparse_s_export_csr(csrA, ret_index, ret_rows, ret_cols,
                                        ret_rows_start, ret_rows_end,
                                        ret_col_index, ret_values),
                "alphasparse_s_export_csr");

  float *y = (float *)alpha_memalign(sizeof(float) * nnz, DEFAULT_ALIGNMENT);
  memcpy(y, *ret_values, sizeof(float) * nnz);
  *ret_values = y;

  printf("ICT To be changed row %d col %d\n", row_index[off], col_index[off]);

  alphasparse_destroy(coo);
  alphasparse_destroy(csrA);

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
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  float *values;
  alpha_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
  ALPHA_INT alpha_off = rand() % nnz;
  
  // return 

  alphasparseIndexBase_t alpha_index;
  ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
  float *alpha_values;

  alpha_set(argc, argv, file, thread_num, &alpha_index, &alpha_rows, &alpha_cols,
          &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values, alpha_off);

  int status = 0;
  if (check) {
#ifdef __MKL__
    sparse_index_base_t mkl_index;
    MKL_INT mkl_off = alpha_off;
    MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
    float *mkl_values;
    mkl_set(argc, argv, file, thread_num, &mkl_index, &mkl_rows, &mkl_cols,
            &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values,
            mkl_off);
    int mkl_nnz = nnz;
    int alpha_nnz = nnz;
    status =
        check_s((float *)mkl_values, mkl_nnz, (float *)alpha_values, alpha_nnz);
    alpha_free(mkl_values);
#endif
  }
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
  alpha_free(alpha_values);
  
  return status;
}