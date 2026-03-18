#include "test_common.h"
/**
 * @brief ict spmmd csr test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static void mkl_spmmd(const int argc, const char *argv[], const char *file,
                      int thread_num, MKL_Complex8 **ret, size_t *ret_size) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  MKL_Complex8 *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  mkl_read_coo_c(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_C = m * m;
  MKL_Complex8 *C = (MKL_Complex8*)alpha_malloc(sizeof(MKL_Complex8) * size_C);
  MKL_INT ldc = m;

  mkl_set_num_threads(thread_num);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);

  sparse_matrix_t coo, csrA, csrB, result;
  mkl_call_exit(mkl_sparse_c_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                                        row_index, col_index, values),
                "mkl_sparse_c_create_coo");
  mkl_call_exit(
      mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
      "mkl_sparse_convert_csr");

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

  mkl_call_exit(mkl_sparse_c_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                                        row_index, col_index, values),
                "mkl_sparse_c_create_coo");
  mkl_call_exit(
      mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csrB),
      "mkl_sparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_sparse_c_spmmd(transA, csrA, csrB, layout, C, ldc);
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_c_spmmd");
  mkl_sparse_destroy(coo);
  mkl_sparse_destroy(csrA);
  mkl_sparse_destroy(csrB);

  *ret = C;
  *ret_size = size_C;
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}
static void alpha_spmmd(const int argc, const char *argv[], const char *file,
                      int thread_num, ALPHA_Complex8 **ret, size_t *ret_size,
                      ALPHA_INT *ret_ldc) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  ALPHA_Complex8 *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  alpha_read_coo_c(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

  size_t size_C = m * m;
  ALPHA_Complex8 *C = (ALPHA_Complex8*)alpha_malloc(sizeof(ALPHA_Complex8) * size_C);
  ALPHA_INT ldc = m;

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, csrA, csrB, result;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);

  alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_c_create_coo");
  alpha_call_exit(
      alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrA),
      "alphasparse_convert_csr");
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
  alpha_call_exit(alphasparse_c_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_c_create_coo");
  alpha_call_exit(
      alphasparse_convert_csr(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &csrB),
      "alphasparse_convert_csr");

  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_call_exit(alphasparse_c_spmmd_plain(transA, csrA, csrB, layout, C, ldc),
                "alphasparse_c_spmmd");
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_c_spmmd");

  *ret = C;
  *ret_ldc = ldc;
  *ret_size = size_C;
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

  ALPHA_Complex8 *alpha_C;
  ALPHA_Complex8 alpha = {0, 0};
  ALPHA_Complex8 beta = {0, 0};
  MKL_Complex8 *mkl_C;
  size_t size_mkl_C, size_alpha_C;

  int status = 0;
  ALPHA_INT ldc;
  alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C, &ldc);

  if (check) {
    mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
    //        status = check_s(float *)mkl_C, 2 * size_mkl_C, (float *)alpha_C, 2
    //        * size_alpha_C);
    status = check_c_l3((ALPHA_Complex8 *)mkl_C, ldc, size_mkl_C, alpha_C, ldc,
                        size_alpha_C, NULL, NULL, 0, alpha_C, ldc, alpha, beta,
                        argc, argv);
    alpha_free(mkl_C);
  }

  alpha_free(alpha_C);
  return status;
}