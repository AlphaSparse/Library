#include "test_common.h"
/**
 * @brief ict spmmd csc test
 * @author HPCRC, ICT
 */

#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

static sparse_status_t alpha_convert_mkl_csc_s(alphasparse_matrix_t src,
                                             sparse_matrix_t *dst,
                                             sparse_index_base_t base,
                                             ALPHA_INT nnz) {
  spmat_csc_s_t *mat = (spmat_csc_s_t *)src->mat;
  // if(base == SPARSE_INDEX_BASE_ONE){
  //     for (ALPHA_INT i = 0; i < mat->rows; i++)
  //     {
  //         mat->cols_end[i] = mat->cols_end[i] + 1;
  //     }
  //     for (ALPHA_INT i = 0; nnz; i++) //nnz
  //     {
  //         mat->row_indx[i] = mat->row_indx[i] + 1;
  //     }
  // }
  sparse_status_t st = mkl_sparse_s_create_csc(
      dst, base, mat->rows, mat->cols, mat->cols_start, mat->cols_end,
      mat->row_indx, (float *)mat->values);
  return st;
}

static void mkl_spmmd(const int argc, const char *argv[], const char *file,
                      int thread_num, float **ret, size_t *ret_size) {
  // 没有考虑column
  sparse_operation_t transA = mkl_args_get_transA(argc, argv);
  sparse_layout_t layout = mkl_args_get_layout(argc, argv);
  MKL_INT mA, kA, nnzA, mB, kB, nnzB;
  MKL_INT *row_indexA, *col_indexA, *row_indexB, *col_indexB;
  float *valuesA, *valuesB;

  const char *fileA = args_get_data_fileA(argc, argv);
  const char *fileB = NULL;
  mkl_read_coo(fileA, &mA, &kA, &nnzA, &row_indexA, &col_indexA, &valuesA);
  if (transA == SPARSE_OPERATION_NON_TRANSPOSE)
    fileB = args_get_data_fileB(argc, argv);
  else
    fileB = args_get_data_fileA(argc, argv);
  mkl_read_coo(fileB, &mB, &kB, &nnzB, &row_indexB, &col_indexB, &valuesB);

  mkl_set_num_threads(thread_num);

  size_t size_C = mA * mA;
  MKL_INT ldc = mA;
  // if(transA == SPARSE_OPERATION_NON_TRANSPOSE){
  //     size_C = mA * kB;
  //     if(layout == SPARSE_LAYOUT_COLUMN_MAJOR) ldc = mA;
  //     else ldc = kB;
  // }
  // else{
  //     size_C = kA * kB;
  //     if(layout == SPARSE_LAYOUT_COLUMN_MAJOR) ldc = kA;
  //     else ldc = kB;
  // }

  float *C = (float*)alpha_malloc(sizeof(float) * size_C);

  sparse_matrix_t result;
  // create cscA
  alphasparse_matrix_t cooA, alpha_cscA;
  sparse_matrix_t cscA;
  alphasparse_s_create_coo(&cooA, ALPHA_SPARSE_INDEX_BASE_ZERO, mA, kA, nnzA,
                          row_indexA, col_indexA, valuesA);
  alphasparse_convert_csc(cooA, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscA);
  alpha_convert_mkl_csc_s(alpha_cscA, &cscA, SPARSE_INDEX_BASE_ZERO, nnzA);
  // mkl_call_exit(mkl_sparse_s_create_coo(&coo, SPARSE_INDEX_BASE_ZERO, m, k,
  // nnz, row_index, col_index, values), "mkl_sparse_s_create_coo");
  // mkl_call_exit(mkl_sparse_convert_csc(coo, SPARSE_OPERATION_NON_TRANSPOSE,
  // &cscA), "mkl_sparse_convert_csc");

  // create cscB
  alphasparse_matrix_t cooB, alpha_cscB;
  sparse_matrix_t cscB;
  alphasparse_s_create_coo(&cooB, ALPHA_SPARSE_INDEX_BASE_ZERO, mB, kB, nnzB,
                          row_indexB, col_indexB, valuesB);
  alphasparse_convert_csc(cooB, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &alpha_cscB);
  alpha_convert_mkl_csc_s(alpha_cscB, &cscB, SPARSE_INDEX_BASE_ZERO, nnzB);

  alpha_timer_t timer;
  alpha_timing_start(&timer);
  mkl_sparse_s_spmmd(transA, cscA, cscB, layout, C, ldc);
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "mkl_sparse_s_spmmd");

  alphasparse_destroy(cooA);
  alphasparse_destroy(cooB);
  mkl_sparse_destroy(cscA);
  mkl_sparse_destroy(cscB);

  *ret = C;
  *ret_size = size_C;
  alpha_free(row_indexA);
  alpha_free(col_indexA);
  alpha_free(valuesA);
  alpha_free(row_indexB);
  alpha_free(col_indexB);
  alpha_free(valuesB);
}

// static void alpha_spmmd(const int argc, const char *argv[], const char *file,
// int thread_num, float **ret, size_t *ret_size)
static void alpha_spmmd(const int argc, const char *argv[], const char *file,
                      int thread_num, float **ret, size_t *ret_size,
                      ALPHA_INT *ret_ldc) {
  ALPHA_INT mA, kA, nnzA, mB, kB, nnzB;
  ALPHA_INT *row_indexA, *col_indexA, *row_indexB, *col_indexB;
  alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
  alphasparse_layout_t layout = alpha_args_get_layout(argc, argv);
  float *valuesA, *valuesB;
  const char *fileA = args_get_data_fileA(argc, argv);
  alpha_read_coo(fileA, &mA, &kA, &nnzA, &row_indexA, &col_indexA, &valuesA);
  const char *fileB = NULL;
  if (transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    fileB = args_get_data_fileB(argc, argv);
  else
    fileB = args_get_data_fileA(argc, argv);
  alpha_read_coo(fileB, &mB, &kB, &nnzB, &row_indexB, &col_indexB, &valuesB);

  size_t size_C = mA * mA;
  ALPHA_INT ldc = mA;
  // if(transA == SPARSE_OPERATION_NON_TRANSPOSE){
  //     size_C = mA * kB;
  //     if(layout == SPARSE_LAYOUT_COLUMN_MAJOR) ldc = kB;
  //     else ldc = kB;
  // }
  // else{
  //     size_C = kA * kB;
  //     if(layout == SPARSE_LAYOUT_COLUMN_MAJOR) ldc = kA;
  //     else ldc = kA;
  // }
  float *C = (float*)alpha_malloc(sizeof(float) * size_C);

  alpha_set_thread_num(thread_num);
  alphasparse_matrix_t coo, cscA, cscB, result;

  // create cscA
  alpha_call_exit(
      alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, mA, kA, nnzA,
                              row_indexA, col_indexA, valuesA),
      "alphasparse_s_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscA),
      "alphasparse_convert_csc");
  alphasparse_destroy(coo);

  // create cscB
  alpha_call_exit(
      alphasparse_s_create_coo(&coo, ALPHA_SPARSE_INDEX_BASE_ZERO, mB, kB, nnzB,
                              row_indexB, col_indexB, valuesB),
      "alphasparse_s_create_coo");
  alpha_call_exit(
      alphasparse_convert_csc(coo, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, &cscB),
      "alphasparse_convert_csc");

  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_call_exit(alphasparse_s_spmmd_plain(transA, cscA, cscB, layout, C, ldc),
                "alphasparse_s_spmmd");
  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alphasparse_s_spmmd");

  *ret = C;
  *ret_ldc = ldc;
  *ret_size = size_C;
  alphasparse_destroy(coo);
  alphasparse_destroy(cscA);
  alphasparse_destroy(cscB);
  alpha_free(row_indexA);
  alpha_free(col_indexA);
  alpha_free(valuesA);
  alpha_free(row_indexB);
  alpha_free(col_indexB);
  alpha_free(valuesB);
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  const char *file = NULL;  // args_get_data_file(argc, argv);
  int thread_num = args_get_thread_num(argc, argv);
  bool check = args_get_if_check(argc, argv);

  float *mkl_C, *alpha_C;
  size_t size_mkl_C, size_alpha_C;
  ALPHA_INT ldc;

  int status = 0;
  // alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C);
  alpha_spmmd(argc, argv, file, thread_num, &alpha_C, &size_alpha_C, &ldc);

  if (check) {
    mkl_spmmd(argc, argv, file, thread_num, &mkl_C, &size_mkl_C);
    status = check_s(mkl_C, size_mkl_C, alpha_C, size_alpha_C);
    status = check_s_l3((float *)mkl_C, ldc, size_mkl_C, alpha_C, ldc, size_alpha_C,
                        NULL, NULL, 0, alpha_C, ldc, 0, 0, argc, argv);

    alpha_free(mkl_C);
  }

  alpha_free(alpha_C);
  return status;
}