#include "test_common.h"
#include <alphasparse.h>
#include <stdio.h>
#ifdef __MKL__
static void mkl_spmm(const int argc, const char *argv[], const char *file, int thread_num,
                     sparse_index_base_t *ret_index, MKL_INT *ret_rows, MKL_INT *ret_cols,
                     MKL_INT **ret_rows_start, MKL_INT **ret_rows_end, MKL_INT **ret_col_index,
                     MKL_Complex8 **ret_values) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex8 *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  mkl_read_coo_c(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

  mkl_set_num_threads(thread_num);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  sparse_matrix_t coo, csrA, csrB, result;
  mkl_call_exit(mkl_sparse_c_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "mkl_sparse_c_create_coo");
  mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_TRANSPOSE, &csrA),
                "mkl_sparse_convert_csc");

  mkl_sparse_destroy(coo);

  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);

  const char *fileB;
  if (transA == SPARSE_OPERATION_NON_TRANSPOSE)
    fileB = args_get_data_fileB(argc, argv);
  else
    fileB = args_get_data_fileA(argc, argv);
  mkl_read_coo_c(fileB, &m, &k, &nnz, &row_index, &col_index, &values);

  mkl_call_exit(mkl_sparse_c_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "mkl_sparse_c_create_coo");
  mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_TRANSPOSE, &csrB),
                "mkl_sparse_convert_csc");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_call_exit(mkl_sparse_spmm(transA, csrA, csrB, &result), "mkl_sparse_spmm");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_spmm");

  mkl_sparse_order(result);

  mkl_call_exit(mkl_sparse_c_export_csr(result, ret_index, ret_rows, ret_cols, ret_rows_start,
                                        ret_rows_end, ret_col_index, ret_values),
                "mkl_sparse_c_export_csc");

  mkl_sparse_destroy(coo);
  mkl_sparse_destroy(csrA);
  mkl_sparse_destroy(csrB);

  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
#endif

static void alpha_spmm(const int argc, const char *argv[], const char *file, int thread_num,
                     alphasparseIndexBase_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT *ret_cols,
                     ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT **ret_col_index,
                     ALPHA_Complex8 **ret_values) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex8 *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  alpha_read_coo_c(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, csrA, csrB, result;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);

  alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_c_create_coo");
  alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
                "alphasparse_convert_csc");
  alphasparse_destroy(coo);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);

  const char *fileB;
  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    fileB = args_get_data_fileB(argc, argv);
  else
    fileB = args_get_data_fileA(argc, argv);
  alpha_read_coo_c(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz, row_index,
                                        col_index, values),
                "alphasparse_c_create_coo");
  alpha_call_exit(alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrB),
                "alphasparse_convert_csc");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_spmm(transA, csrA, csrB, &result), "alphasparse_spmm");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_spmm");

  alpha_call_exit(alphasparse_c_export_csc(result, ret_index, ret_rows, ret_cols, ret_rows_start,
                                        ret_rows_end, ret_col_index, ret_values),
                "alphasparse_c_export_csc");

  alphasparse_destroy(coo);
  alphasparse_destroy(csrA);
  alphasparse_destroy(csrB);

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

  // return
#ifdef __MKL__
  sparse_index_base_t mkl_index;
  MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
  MKL_Complex8 *mkl_values;
#endif

  alphasparseIndexBase_t alpha_index;
  ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
  ALPHA_Complex8 *alpha_values;
  ALPHA_Complex8 alpha = {0.f, 0.f};
  ALPHA_Complex8 beta = {0.f, 0.f};

  alpha_spmm(argc, argv, file, thread_num, &alpha_index, &alpha_rows, &alpha_cols, &alpha_rows_start,
           &alpha_rows_end, &alpha_col_index, &alpha_values);

  int status = 0;
  if (check) {
#ifdef __MKL__
    mkl_spmm(argc, argv, file, thread_num, &mkl_index, &mkl_rows, &mkl_cols, &mkl_rows_start,
             &mkl_rows_end, &mkl_col_index, &mkl_values);
    int mkl_nnz = mkl_rows_end[mkl_rows - 1];
    int alpha_nnz = alpha_rows_end[alpha_rows - 1];
    status = check_s((float *)mkl_values, mkl_nnz*2, (float *)alpha_values, alpha_nnz*2);
#endif

    // status = check_c_l3((float *)mkl_values, 0, mkl_nnz, alpha_values, 0, alpha_nnz,
    //                     alpha_col_index, NULL, 0, alpha_values, 0, alpha, beta,
    //                     argc, argv);
  }

  return status;
}