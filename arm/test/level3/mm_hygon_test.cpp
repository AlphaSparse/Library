#include <alphasparse.h>
#include <stdio.h>

#include "test_common.h"
const int block_size = 4;

void alpha_mm(matrix_data_t *matrix_data, alpha_common_args_t *common_arg, const char *x_char, int ldx,
            char *icty_char, int ldy, const char *alpha_char, const char *beta_char) {
  // 设置使用线程数
  alpha_set_thread_num(common_arg->thread_num);

  alphasparse_matrix_t cooA, compute_matrix;
  alpha_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix = cooA;
  } else {
    alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, common_arg->layout, cooA, &compute_matrix,
                              block_size, block_size);
  }

  alpha_timer_t timer;
  double total_time = 0.;
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    if (common_arg->warm) {
      alpha_call_exit(
          alphasparse_s_mm(common_arg->transA, *((float *)alpha_char), compute_matrix,
                          common_arg->alpha_descr, common_arg->layout, (float *)x_char,
                          common_arg->columns, ldx, *((float *)beta_char), (float *)icty_char, ldy),
          "alphasparse_s_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(
          alphasparse_s_mm(common_arg->transA, *((float *)alpha_char), compute_matrix,
                          common_arg->alpha_descr, common_arg->layout, (float *)x_char,
                          common_arg->columns, ldx, *((float *)beta_char), (float *)icty_char, ldy),
          "alphasparse_s_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    if (common_arg->warm) {
      alpha_call_exit(alphasparse_d_mm(common_arg->transA, *((double *)alpha_char), compute_matrix,
                                    common_arg->alpha_descr, common_arg->layout, (double *)x_char,
                                    common_arg->columns, ldx, *((double *)beta_char),
                                    (double *)icty_char, ldy),
                    "alphasparse_d_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_d_mm(common_arg->transA, *((double *)alpha_char), compute_matrix,
                                    common_arg->alpha_descr, common_arg->layout, (double *)x_char,
                                    common_arg->columns, ldx, *((double *)beta_char),
                                    (double *)icty_char, ldy),
                    "alphasparse_d_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    if (common_arg->warm) {
      alpha_call_exit(alphasparse_c_mm(common_arg->transA, *((ALPHA_Complex8 *)alpha_char),
                                    compute_matrix, common_arg->alpha_descr, common_arg->layout,
                                    (ALPHA_Complex8 *)x_char, common_arg->columns, ldx,
                                    *((ALPHA_Complex8 *)beta_char), (ALPHA_Complex8 *)icty_char, ldy),
                    "alphasparse_c_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_c_mm(common_arg->transA, *((ALPHA_Complex8 *)alpha_char),
                                    compute_matrix, common_arg->alpha_descr, common_arg->layout,
                                    (ALPHA_Complex8 *)x_char, common_arg->columns, ldx,
                                    *((ALPHA_Complex8 *)beta_char), (ALPHA_Complex8 *)icty_char, ldy),
                    "alphasparse_c_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    if (common_arg->warm) {
      alpha_call_exit(alphasparse_z_mm(common_arg->transA, *((ALPHA_Complex16 *)alpha_char),
                                    compute_matrix, common_arg->alpha_descr, common_arg->layout,
                                    (ALPHA_Complex16 *)x_char, common_arg->columns, ldx,
                                    *((ALPHA_Complex16 *)beta_char), (ALPHA_Complex16 *)icty_char, ldy),
                    "alphasparse_z_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(alphasparse_z_mm(common_arg->transA, *((ALPHA_Complex16 *)alpha_char),
                                    compute_matrix, common_arg->alpha_descr, common_arg->layout,
                                    (ALPHA_Complex16 *)x_char, common_arg->columns, ldx,
                                    *((ALPHA_Complex16 *)beta_char), (ALPHA_Complex16 *)icty_char, ldy),
                    "alphasparse_z_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  }
  printf("%s time : %lf[ms]\n", "alphasparse_mm", (total_time / common_arg->iter) * 1000);

  alphasparse_destroy(cooA);
  if (common_arg->format != ALPHA_SPARSE_FORMAT_COO) alphasparse_destroy(compute_matrix);
}

// void alpha_mm_plain(matrix_data_t *matrix_data, alpha_common_args_t *common_arg, const char *x_char,
//                   int ldx, char *icty_char, int ldy, const char *alpha_char,
//                   const char *beta_char) {
//   // 设置使用线程数

//   alphasparse_matrix_t cooA, compute_matrix;
//   alpha_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
//   if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
//     compute_matrix = cooA;
//   } else {
//     alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, cooA, &compute_matrix,
//                               block_size, block_size);
//   }

//   alpha_timer_t timer;
//   double total_time = 0.;
//   alpha_set_thread_num(1);
//   if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
//     if (common_arg->warm) {
//       alpha_call_exit(alphasparse_s_mm_plain(common_arg->transA, *((float *)alpha_char),
//                                           compute_matrix, common_arg->alpha_descr, common_arg->layout,
//                                           (float *)x_char, common_arg->columns, ldx,
//                                           *((float *)beta_char), (float *)icty_char, ldy),
//                     "alphasparse_s_mm_plain");
//     }
//     alpha_timing_start(&timer);
//     for (int i = 0; i < common_arg->iter; i++) {
//       alpha_call_exit(alphasparse_s_mm_plain(common_arg->transA, *((float *)alpha_char),
//                                           compute_matrix, common_arg->alpha_descr, common_arg->layout,
//                                           (float *)x_char, common_arg->columns, ldx,
//                                           *((float *)beta_char), (float *)icty_char, ldy),
//                     "alphasparse_s_mm_plain");
//     }
//     alpha_timing_end(&timer);
//     total_time = alpha_timing_elapsed_time(&timer);
//   } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
//     if (common_arg->warm) {
//       alpha_call_exit(alphasparse_d_mm_plain(common_arg->transA, *((double *)alpha_char),
//                                           compute_matrix, common_arg->alpha_descr, common_arg->layout,
//                                           (double *)x_char, common_arg->columns, ldx,
//                                           *((double *)beta_char), (double *)icty_char, ldy),
//                     "alphasparse_d_mm_plain");
//     }
//     alpha_timing_start(&timer);
//     for (int i = 0; i < common_arg->iter; i++) {
//       alpha_call_exit(alphasparse_d_mm_plain(common_arg->transA, *((double *)alpha_char),
//                                           compute_matrix, common_arg->alpha_descr, common_arg->layout,
//                                           (double *)x_char, common_arg->columns, ldx,
//                                           *((double *)beta_char), (double *)icty_char, ldy),
//                     "alphasparse_d_mm_plain");
//     }
//     alpha_timing_end(&timer);
//     total_time = alpha_timing_elapsed_time(&timer);
//   } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
//     if (common_arg->warm) {
//       alpha_call_exit(
//           alphasparse_c_mm_plain(common_arg->transA, *((ALPHA_Complex8 *)alpha_char), compute_matrix,
//                                 common_arg->alpha_descr, common_arg->layout, (ALPHA_Complex8 *)x_char,
//                                 common_arg->columns, ldx, *((ALPHA_Complex8 *)beta_char),
//                                 (ALPHA_Complex8 *)icty_char, ldy),
//           "alphasparse_c_mm_plain");
//     }
//     alpha_timing_start(&timer);
//     for (int i = 0; i < common_arg->iter; i++) {
//       alpha_call_exit(
//           alphasparse_c_mm_plain(common_arg->transA, *((ALPHA_Complex8 *)alpha_char), compute_matrix,
//                                 common_arg->alpha_descr, common_arg->layout, (ALPHA_Complex8 *)x_char,
//                                 common_arg->columns, ldx, *((ALPHA_Complex8 *)beta_char),
//                                 (ALPHA_Complex8 *)icty_char, ldy),
//           "alphasparse_c_mm_plain");
//     }
//     alpha_timing_end(&timer);
//     total_time = alpha_timing_elapsed_time(&timer);
//   } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
//     if (common_arg->warm) {
//       alpha_call_exit(
//           alphasparse_z_mm_plain(common_arg->transA, *((ALPHA_Complex16 *)alpha_char), compute_matrix,
//                                 common_arg->alpha_descr, common_arg->layout, (ALPHA_Complex16 *)x_char,
//                                 common_arg->columns, ldx, *((ALPHA_Complex16 *)beta_char),
//                                 (ALPHA_Complex16 *)icty_char, ldy),
//           "alphasparse_z_mm_plain");
//     }
//     alpha_timing_start(&timer);
//     for (int i = 0; i < common_arg->iter; i++) {
//       alpha_call_exit(
//           alphasparse_z_mm_plain(common_arg->transA, *((ALPHA_Complex16 *)alpha_char), compute_matrix,
//                                 common_arg->alpha_descr, common_arg->layout, (ALPHA_Complex16 *)x_char,
//                                 common_arg->columns, ldx, *((ALPHA_Complex16 *)beta_char),
//                                 (ALPHA_Complex16 *)icty_char, ldy),
//           "alphasparse_z_mm_plain");
//     }
//     alpha_timing_end(&timer);
//     total_time = alpha_timing_elapsed_time(&timer);
//   }
//   printf("%s time : %lf[ms]\n", "alphasparse_mm_plain", (total_time / common_arg->iter) * 1000);
//   alphasparse_destroy(cooA);
//   if (common_arg->format != ALPHA_SPARSE_FORMAT_COO) alphasparse_destroy(compute_matrix);
// }
#ifdef __MKL__
void mkl_mm(matrix_data_t *matrix_data, alpha_common_args_t *common_arg, const char *x_char, int ldx,
            char *icty_char, int ldy, const char *alpha_char, const char *beta_char) {
  // 设置使用线程数
  sparse_matrix_t cooA, compute_matrix;
  mkl_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix = cooA;
  } else {
    mkl_convert_matrix_wapper(common_arg->format, common_arg->mkl_descr, common_arg->mkl_layout, cooA, &compute_matrix,
                              block_size, block_size);
  }

  alpha_timer_t timer;
  double total_time = 0.;
  mkl_set_num_threads(common_arg->thread_num);
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    if (common_arg->warm)
      mkl_call_exit(
          mkl_sparse_s_mm(common_arg->mkl_transA, *((float *)alpha_char), compute_matrix,
                          common_arg->mkl_descr, common_arg->mkl_layout, (float *)x_char,
                          common_arg->columns, ldx, *((float *)beta_char), (float *)icty_char, ldy),
          "mkl_sparse_s_mm");
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(
          mkl_sparse_s_mm(common_arg->mkl_transA, *((float *)alpha_char), compute_matrix,
                          common_arg->mkl_descr, common_arg->mkl_layout, (float *)x_char,
                          common_arg->columns, ldx, *((float *)beta_char), (float *)icty_char, ldy),
          "mkl_sparse_s_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    if (common_arg->warm) {
      mkl_call_exit(mkl_sparse_d_mm(common_arg->mkl_transA, *((double *)alpha_char), compute_matrix,
                                    common_arg->mkl_descr, common_arg->mkl_layout, (double *)x_char,
                                    common_arg->columns, ldx, *((double *)beta_char),
                                    (double *)icty_char, ldy),
                    "mkl_sparse_d_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(mkl_sparse_d_mm(common_arg->mkl_transA, *((double *)alpha_char), compute_matrix,
                                    common_arg->mkl_descr, common_arg->mkl_layout, (double *)x_char,
                                    common_arg->columns, ldx, *((double *)beta_char),
                                    (double *)icty_char, ldy),
                    "mkl_sparse_d_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    if (common_arg->warm) {
      mkl_call_exit(mkl_sparse_c_mm(common_arg->mkl_transA, *((MKL_Complex8 *)alpha_char),
                                    compute_matrix, common_arg->mkl_descr, common_arg->mkl_layout,
                                    (MKL_Complex8 *)x_char, common_arg->columns, ldx,
                                    *((MKL_Complex8 *)beta_char), (MKL_Complex8 *)icty_char, ldy),
                    "mkl_sparse_c_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(mkl_sparse_c_mm(common_arg->mkl_transA, *((MKL_Complex8 *)alpha_char),
                                    compute_matrix, common_arg->mkl_descr, common_arg->mkl_layout,
                                    (MKL_Complex8 *)x_char, common_arg->columns, ldx,
                                    *((MKL_Complex8 *)beta_char), (MKL_Complex8 *)icty_char, ldy),
                    "mkl_sparse_c_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    if (common_arg->warm) {
      mkl_call_exit(mkl_sparse_z_mm(common_arg->mkl_transA, *((MKL_Complex16 *)alpha_char),
                                    compute_matrix, common_arg->mkl_descr, common_arg->mkl_layout,
                                    (MKL_Complex16 *)x_char, common_arg->columns, ldx,
                                    *((MKL_Complex16 *)beta_char), (MKL_Complex16 *)icty_char, ldy),
                    "mkl_sparse_z_mm");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      mkl_call_exit(mkl_sparse_z_mm(common_arg->mkl_transA, *((MKL_Complex16 *)alpha_char),
                                    compute_matrix, common_arg->mkl_descr, common_arg->mkl_layout,
                                    (MKL_Complex16 *)x_char, common_arg->columns, ldx,
                                    *((MKL_Complex16 *)beta_char), (MKL_Complex16 *)icty_char, ldy),
                    "mkl_sparse_z_mm");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  }
  printf("%s time : %lf[ms]\n", "mkl_sparse_mm", (total_time / common_arg->iter) * 1000);
  mkl_sparse_destroy(cooA);
  if (common_arg->format != ALPHA_SPARSE_FORMAT_COO) mkl_sparse_destroy(compute_matrix);
}
#endif

int main(int argc, const char *argv[]) {
  // args
  alpha_common_args_t common_arg;
  // read_coo
  matrix_data_t matrix_data;
  args_help(argc, argv);

  alpha_timer_t timer;
  // init args
  parse_args_and_initialize(argc, argv, &common_arg);
  alpha_timing_start(&timer);
  alpha_read_coo_wrapper(&matrix_data, &common_arg, FILE_SOURCE, block_size);
  alpha_timing_end(&timer);
  double time_elapsed = alpha_timing_elapsed_time(&timer);
  printf("io elapesd %f [ms]\n", time_elapsed * 1e3);
  // args
  int ldx, ldy;
  ldx = common_arg.columns, ldy = common_arg.columns;
  ALPHA_INT rowsx = matrix_data.k, rowsy = matrix_data.m;
  if (common_arg.transA == ALPHA_SPARSE_OPERATION_TRANSPOSE ||
      common_arg.transA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    rowsx = matrix_data.m;
    rowsy = matrix_data.k;
  }
  if (common_arg.layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    ldx = rowsx;
    ldy = rowsy;
  }
  ALPHA_INT64 sizex = rowsx * common_arg.columns;
  ALPHA_INT64 sizey = rowsy * common_arg.columns;
  char *x_char;
  char *icty_char;
  char *icty_plain_char;
  char *alpha_alpha_char;
  char *alpha_beta_char;

#ifdef __MKL__
  char *mkl_alpha_char;
  char *mkl_beta_char;
#endif

  alpha_alpha_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);
  alpha_beta_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);

#ifdef __MKL__
  mkl_alpha_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);
  mkl_beta_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);
#endif
  alpha_initialize_alpha_beta(alpha_alpha_char, alpha_beta_char, common_arg.data_type);
#ifdef __MKL__
  mkl_initialize_alpha_beta(mkl_alpha_char, mkl_beta_char, common_arg.data_type);
#endif

  malloc_random_fill(common_arg.data_type, (void **)&x_char, sizex, 0);  // x,y
  malloc_random_fill(common_arg.data_type, (void **)&icty_char, sizey,
                     1);  // x,y
  // printf("thread_num : %d\n",thread_num);
  alpha_mm(&matrix_data, &common_arg, x_char, ldx, icty_char, ldy, alpha_alpha_char, alpha_beta_char);
  if (common_arg.check) {
    malloc_random_fill(common_arg.data_type, (void **)&icty_plain_char, sizey,
                       1);  // x,y
#ifdef __MKL__
    mkl_mm(&matrix_data, &common_arg, x_char, ldx, icty_plain_char, ldy, mkl_alpha_char,
                 mkl_beta_char);
// #else 
//     alpha_mm_plain(&matrix_data, &common_arg, x_char, ldx, icty_plain_char, ldy, alpha_alpha_char,
//                  alpha_beta_char);
#endif
    check_arm(common_arg.data_type, icty_char, sizey, icty_plain_char);
    alpha_free(icty_plain_char);
  }
  alpha_free(x_char);
  alpha_free(icty_char);

  destory_matrix_data(&matrix_data);
  return 0;
}
