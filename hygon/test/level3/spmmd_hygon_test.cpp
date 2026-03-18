#include <alphasparse.h>
#include <stdio.h>

#include "test_common.h"
const int block_size = 4;

// matrix_data_lhs == matrix_data_rhs when op == transpose
void alpha_spmmd(matrix_data_t *matrix_data_lhs, matrix_data_t *matrix_data_rhs,
               alpha_common_args_t *common_arg, char **ret, size_t *ret_size) {
  // 设置使用线程数
  alpha_set_thread_num(common_arg->thread_num);
  int rowc = matrix_data_lhs->m;
  int colc = matrix_data_lhs->k;

  if (common_arg->transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      common_arg->transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rowc = matrix_data_lhs->m;
    colc = matrix_data_lhs->m;
  }

  ALPHA_INT ldc = colc;
  if (common_arg->layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldc = rowc;
  }
  size_t size_C = rowc * colc;
  char *C_char = (char *)alpha_malloc(bytes_type[common_arg->data_type] * size_C);
  *ret = C_char;
  *ret_size = size_C;
  alphasparse_matrix_t coo_lhs, coo_rhs, compute_matrix_lhs, compute_matrix_rhs;
  alpha_create_coo_wapper(matrix_data_lhs, common_arg->data_type, &coo_lhs);
  alpha_create_coo_wapper(matrix_data_rhs, common_arg->data_type, &coo_rhs);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix_lhs = coo_lhs;
    compute_matrix_rhs = coo_rhs;
  } else {
    alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, common_arg->layout, coo_lhs,
                              &compute_matrix_lhs, block_size, block_size);
    alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, common_arg->layout, coo_rhs,
                              &compute_matrix_rhs, block_size, block_size);
  }

  alpha_timer_t timer;
  double total_time = 0.;
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_s_spmmd(common_arg->transA, compute_matrix_lhs, compute_matrix_rhs,
                                       common_arg->layout, (float *)C_char, ldc),
                    "alphasparse_s_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_d_spmmd(common_arg->transA, compute_matrix_lhs, compute_matrix_rhs,
                                       common_arg->layout, (double *)C_char, ldc),
                    "alphasparse_d_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_c_spmmd(common_arg->transA, compute_matrix_lhs, compute_matrix_rhs,
                                       common_arg->layout, (ALPHA_Complex8 *)C_char, ldc),
                    "alphasparse_c_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_z_spmmd(common_arg->transA, compute_matrix_lhs, compute_matrix_rhs,
                                       common_arg->layout, (ALPHA_Complex16 *)C_char, ldc),
                    "alphasparse_z_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  }

  printf("%s time : %lf[ms]\n", "alphasparse_spmmd", (total_time / common_arg->iter) * 1000);

  alphasparse_destroy(compute_matrix_lhs);
  alphasparse_destroy(compute_matrix_rhs);
}
#ifdef __MKL__
void mkl_spmmd(matrix_data_t *matrix_data_lhs, matrix_data_t *matrix_data_rhs,
               alpha_common_args_t *common_arg, char **ret, size_t *ret_size) {
  // 设置使用线程数
  int rowc = matrix_data_lhs->m;
  int colc = matrix_data_lhs->k;

  if (common_arg->transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      common_arg->transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rowc = matrix_data_lhs->m;
    colc = matrix_data_lhs->m;
  }

  ALPHA_INT ldc = colc;
  if (common_arg->layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldc = rowc;
  }

  size_t size_C = rowc * colc;
  char *C_char = (char *)alpha_malloc(bytes_type[common_arg->data_type] * size_C);
  *ret = C_char;
  *ret_size = size_C;
  sparse_matrix_t coo_lhs, coo_rhs, compute_matrix_lhs, compute_matrix_rhs;
  mkl_create_coo_wapper(matrix_data_lhs, common_arg->data_type, &coo_lhs);
  mkl_create_coo_wapper(matrix_data_rhs, common_arg->data_type, &coo_rhs);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix_lhs = coo_lhs;
    compute_matrix_rhs = coo_rhs;
  } else {
    mkl_convert_matrix_wapper(common_arg->format, common_arg->mkl_descr, common_arg->mkl_layout, coo_lhs,
                              &compute_matrix_lhs, block_size, block_size);
    mkl_convert_matrix_wapper(common_arg->format, common_arg->mkl_descr, common_arg->mkl_layout, coo_rhs,
                              &compute_matrix_rhs, block_size, block_size);
  }

  alpha_set_thread_num(1);
  alpha_timer_t timer;
  double total_time = 0.;
  mkl_set_num_threads(common_arg->thread_num);
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(
          mkl_sparse_s_spmmd(common_arg->mkl_transA, compute_matrix_lhs, compute_matrix_rhs,
                             common_arg->mkl_layout, (float *)C_char, ldc),
          "mkl_sparse_s_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(
          mkl_sparse_d_spmmd(common_arg->mkl_transA, compute_matrix_lhs, compute_matrix_rhs,
                             common_arg->mkl_layout, (double *)C_char, ldc),
          "mkl_sparse_d_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(
          mkl_sparse_c_spmmd(common_arg->mkl_transA, compute_matrix_lhs, compute_matrix_rhs,
                             common_arg->mkl_layout, (MKL_Complex8 *)C_char, ldc),
          "mkl_sparse_c_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(
          mkl_sparse_z_spmmd(common_arg->mkl_transA, compute_matrix_lhs, compute_matrix_rhs,
                             common_arg->mkl_layout, (MKL_Complex16 *)C_char, ldc),
          "mkl_sparse_z_spmmd");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  }
  printf("%s time : %lf[ms]\n", "mkl_sparse_spmmd", (total_time / common_arg->iter) * 1000);

  mkl_sparse_destroy(compute_matrix_lhs);
  mkl_sparse_destroy(compute_matrix_rhs);
}
#endif
int main(int argc, const char *argv[]) {
  // args
  alpha_common_args_t common_arg;
  // read_coo
  matrix_data_t matrix_dataA;
  matrix_data_t matrix_dataB;
  args_help(argc, argv);

  // init args
  parse_args_and_initialize(argc, argv, &common_arg);
  alpha_read_coo_wrapper(&matrix_dataA, &common_arg, FILE_SOURCE_A, 0);
  alpha_read_coo_wrapper(&matrix_dataB, &common_arg, FILE_SOURCE_B, 0);

  // ret matrix
  char *ret;
  char *ret_plain;

  size_t ret_size;
  size_t ret_size_plain;

  alpha_spmmd(&matrix_dataA, &matrix_dataB, &common_arg, (char **)&ret, &ret_size);
  if (common_arg.check) {
#ifdef __MKL__
    mkl_spmmd(&matrix_dataA, &matrix_dataB, &common_arg, (char **)&ret_plain,
                    &ret_size_plain);
    check_arm(common_arg.data_type, ret, ret_size, ret_plain);
    alpha_free(ret_plain);
#endif
  }
  alpha_free(ret);

  destory_matrix_data(&matrix_dataA);
  destory_matrix_data(&matrix_dataB);
  return 0;
}
