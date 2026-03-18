#include "test_common.h"
/**
 * @brief ict spmm csc test
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static sparse_status_t alpha_convert_mkl_csc_c(alphasparse_matrix_t src,
                                             sparse_matrix_t *dst) {
  spmat_csc_s_t *mat = (spmat_csc_s_t *)src->mat;
  sparse_status_t st = mkl_sparse_s_create_csc(
      dst, SPARSE_INDEX_BASE_ZERO, mat->rows, mat->cols, mat->cols_start,
      mat->cols_end, mat->row_indx, (float *)mat->values);
  return st;
}

static void mkl_spmm(const int argc, const char *argv[], const char *file,
                     int thread_num, sparse_index_base_t *ret_index,
                     MKL_INT *ret_rows, MKL_INT *ret_cols,
                     MKL_INT **ret_rows_start, MKL_INT **ret_rows_end,
                     MKL_INT **ret_col_index, float **ret_values) {
  MKL_INT m, k, nnz;
  MKL_INT *row_index, *col_index;
  float *values;
  // mkl_read_coo(file, &m, &k, &nnz, &row_index, &col_index, &values);
  // alpha_fill_random_s(values, 1, nnz);

  // mkl_set_num_threads(thread_num);
  // sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  // sparse_matrix_t coo, csr, result;
  // mkl_call_exit(mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k,
  // nnz, row_index, col_index, values), "mkl_sparse_s_create_coo"); if(transA
  // == SPARSE_OPERATION_NON_TRANSPOSE)
  //     {mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_TRANSPOSE,
  //     &csr), "mkl_sparse_convert_csr");}
  // else
  //     mkl_call_exit(mkl_sparse_convert_csr(coo,
  //     SPARSE_OPERATION_NON_TRANSPOSE, &csr), "mkl_sparse_convert_csr");

  sparse_matrix_t result;
  // create cscA
  const char *fileA = args_get_data_fileA(argc, argv);
  alpha_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);

  mkl_set_num_threads(thread_num);
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);

  // sparse_matrix_t coo, csrA, csrB, result;
  // mkl_call_exit(mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k,
  // nnz, row_index, col_index, values), "mkl_sparse_s_create_coo");
  // mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE,
  // &csrA), "mkl_sparse_convert_csr");
  alphasparse_matrix_t cooA, alpha_cscA;
  sparse_matrix_t cscA;
  alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                          row_index, col_index, values);
  alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
  alpha_convert_mkl_csc_c(alpha_cscA, &cscA);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);

  // create cscB
  const char *fileB = NULL;
  if (transA == SPARSE_OPERATION_NON_TRANSPOSE)
    fileB = args_get_data_fileB(argc, argv);
  else
    fileB = args_get_data_fileA(argc, argv);
  alpha_read_coo(fileB, &m, &k, &nnz, &row_index, &col_index, &values);

  // mkl_call_exit(mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k,
  // nnz, row_index, col_index, values), "mkl_sparse_s_create_coo");
  // mkl_call_exit(mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE,
  // &csrB), "mkl_sparse_convert_csr");
  alphasparse_matrix_t cooB, alpha_cscB;
  sparse_matrix_t cscB;
  alphasparse_s_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k, nnz,
                          row_index, col_index, values);
  alphasparse_convert_csc(cooB, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscB);
  alpha_convert_mkl_csc_c(alpha_cscB, &cscB);

  // spmm
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  mkl_call_exit(mkl_sparse_spmm(transA, cscA, cscB, &result),
                "mkl_sparse_spmm");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_spmm");

  mkl_sparse_order(result);

  mkl_call_exit(mkl_sparse_s_export_csc(result, ret_index, ret_rows, ret_cols,
                                        ret_rows_start, ret_rows_end,
                                        ret_col_index, ret_values),
                "mkl_sparse_d_export_csc");

  alphasparse_destroy(cooA);
  alphasparse_destroy(cooB);
  mkl_sparse_destroy(cscA);
  mkl_sparse_destroy(cscB);

  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);
}

// static void alpha_spmm(const int argc, const char *argv[], const char *file,
// int thread_num, alphasparseIndexBase_t *ret_index, ALPHA_INT *ret_rows, ALPHA_INT
// *ret_cols, ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end, ALPHA_INT
// **ret_col_index, float **ret_values)
static void alpha_spmm(const int argc, const char *argv[], const char *file,
                     int thread_num, alphasparseIndexBase_t *ret_index,
                     ALPHA_INT *ret_rows, ALPHA_INT *ret_cols,
                     ALPHA_INT **ret_rows_start, ALPHA_INT **ret_rows_end,
                     ALPHA_INT **ret_col_index, float **ret_values) {
  ALPHA_INT m, k, nnz;
  ALPHA_INT *row_index, *col_index;
  float *values;
  const char *fileA = args_get_data_fileA(argc, argv);
  alpha_read_coo(fileA, &m, &k, &nnz, &row_index, &col_index, &values);
  // alpha_fill_random_s(values, 1, nnz);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, cscA, cscB, result;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);

  alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_s_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA),
      "alphasparse_convert_csc");
  alphasparse_destroy(coo);
  alpha_free(row_index);
  alpha_free(col_index);
  alpha_free(values);

  const char *fileB = NULL;
  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    fileB = args_get_data_fileB(argc, argv);
  else
    fileB = args_get_data_fileA(argc, argv);
  alpha_read_coo(fileB, &m, &k, &nnz, &row_index, &col_index, &values);
  alpha_call_exit(alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                        nnz, row_index, col_index, values),
                "alphasparse_s_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB),
      "alphasparse_convert_csc");

  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_spmm_plain(transA, cscA, cscB, &result),
                "alphasparse_spmm_plain");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_spmm_plain");

  alpha_call_exit(alphasparse_s_export_csc(result, ret_index, ret_rows, ret_cols,
                                        ret_rows_start, ret_rows_end,
                                        ret_col_index, ret_values),
                "alphasparse_s_export_csc");

  alphasparse_destroy(coo);
  alphasparse_destroy(cscA);
  alphasparse_destroy(cscB);

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
  sparse_index_base_t mkl_index;
  MKL_INT mkl_rows, mkl_cols, *mkl_rows_start, *mkl_rows_end, *mkl_col_index;
  float *mkl_values;

  alphasparseIndexBase_t alpha_index;
  ALPHA_INT alpha_rows, alpha_cols, *alpha_rows_start, *alpha_rows_end, *alpha_col_index;
  float *alpha_values;

  alpha_spmm(argc, argv, file, thread_num, &alpha_index, &alpha_rows, &alpha_cols,
           &alpha_rows_start, &alpha_rows_end, &alpha_col_index, &alpha_values);

  int alpha_nnz = alpha_rows_end[alpha_cols - 1];
  ALPHA_INT *col_index = (ALPHA_INT*)malloc(sizeof(ALPHA_INT) * alpha_nnz);

  for (ALPHA_INT i = 0, cnt = 0; i < alpha_cols; i++) {
    for (ALPHA_INT j = alpha_rows_start[i]; j < alpha_rows_end[i]; j++) {
      col_index[cnt++] = i;
    }
  }

  int status = 0;
  if (check) {
    mkl_spmm(argc, argv, file, thread_num, &mkl_index, &mkl_rows, &mkl_cols,
             &mkl_rows_start, &mkl_rows_end, &mkl_col_index, &mkl_values);
    int mkl_nnz = mkl_rows_end[mkl_cols - 1];
    int alpha_nnz = alpha_rows_end[alpha_cols - 1];
    // status = check_s(mkl_values, mkl_nnz, alpha_values, alpha_nnz);
    float zero = 0.f;
    status =
        check_s_l3((float *)mkl_values, 0, mkl_nnz, alpha_values, 0, alpha_nnz,
                   col_index, NULL, 0, alpha_values, 0, zero, zero, argc, argv);
  }

  return status;
}