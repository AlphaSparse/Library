#include <alphasparse.h>
#include <stdio.h>
#include <vector>
#include <algorithm>  
#include <numeric>    
#include <string.h>
#include "../include/test_common.h"
// sigma = 1 : regress to sell
// C = 1 : regress to csr
int param0 = 8;    // block_row or C
int param1 = 256;  // block_col or Sigma
double get_avg_time(std::vector<double> times) {
  std::sort(times.begin(), times.end());
  int len = times.size();
  double median;
  if (len % 2 == 0) {
    median = (times[len / 2 - 1] + times[len / 2]) / 2;
  } else {
    median = times[len / 2];
  }
  double lim_l = median * 0.9;
  double lim_r = median * 1.1;
  int cnt = 0;
  double sum = 0;
  for (double x : times) {
    if (x >= lim_l && x <= lim_r) {
      sum += x;
      cnt++;
    }
  }
  if (cnt == 0) {
    return median;
  }
  return sum / cnt;
}
static void mkl_mv_dia(matrix_data_t *matrix_data,
                       alpha_common_args_t *common_arg,
                       const char *x_char,
                       char *icty_char,
                       const char *alpha_alpha_char,
                       const char *alpha_beta_char)
{
    alpha_set_thread_num(common_arg->thread_num);
    alphasparse_matrix_t cooA, compute_matrix;
    alpha_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
    if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
        compute_matrix = cooA;
    } else {
        alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, common_arg->layout, 
                                    cooA, &compute_matrix, param0, param1);
    }

    if (common_arg->format != ALPHA_SPARSE_FORMAT_DIA) {
        fprintf(stderr, "Error: compute_matrix is not in DIA format\n");
        exit(EXIT_FAILURE);
    }

    const char *transa = (common_arg->transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) ? "N" : "T";
    MKL_INT m = compute_matrix->mat->rows;          
    MKL_INT k = compute_matrix->mat->cols;          
    const double *alpha = (const double*)alpha_alpha_char; 
    const char matdescra[6] = {'G', 'L', 'N', 'F', '\0', '\0'};  
    
    const double *val = (const double*)compute_matrix->mat->val_data; 
    MKL_INT lval = compute_matrix->mat->lval;       
    const MKL_INT *idiag = compute_matrix->mat->dis_data; 
    MKL_INT ndiag = compute_matrix->mat->ndiag;    
    const double *x = (const double*)x_char;     
    const double *beta = (const double*)alpha_beta_char;   
    double *y = (double*)icty_char;  

    for (int i = 0; i < 20; i++){
        mkl_ddiamv(
            transa,
            &m,
            &k,
            alpha,
            matdescra,
            val,
            &lval,
            idiag,
            &ndiag,
            x,
            beta,
            y
        );
    }
    alpha_timer_t timer;
    double onetime = 0.;
    std::vector<double> times;
    
    for (int i = 0; i < common_arg->iter; i++) {
        alpha_timing_start(&timer);
        mkl_ddiamv(
              transa,
              &m,
              &k,
              alpha,
              matdescra,
              val,
              &lval,
              idiag,
              &ndiag,
              x,
              beta,
              y
          );
        alpha_timing_end(&timer);
        onetime = alpha_timing_elapsed_time(&timer);
        times.push_back(onetime);
    }
    
    
    printf("%s_time : %lf\n", "mkl_dia",
            get_avg_time(times) * 1000);
    // if (common_arg->format != ALPHA_SPARSE_FORMAT_COO) {
    //     alpha_destroy_matrix(compute_matrix);
    // }
    // alpha_destroy_matrix(cooA);
}

static void alpha_mv_autoSelect(matrix_data_t *matrix_data,
                     alpha_common_args_t *common_arg, const char *x_char,
                     char *icty_char, const char *alpha_alpha_char,
                     const char *alpha_beta_char) {
  alpha_set_thread_num(common_arg->thread_num);
  alphasparse_matrix_t cooA, compute_matrix;
  alpha_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix = cooA;
  } else {
    alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, common_arg->layout, 
                                cooA, &compute_matrix, param0, param1);
  }

}
static void alpha_mv(matrix_data_t *matrix_data,
                     alpha_common_args_t *common_arg, const char *x_char,
                     char *icty_char, const char *alpha_alpha_char,
                     const char *alpha_beta_char) {
  alpha_set_thread_num(common_arg->thread_num);
  alphasparse_matrix_t cooA, compute_matrix;
  alpha_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix = cooA;
  } else {
    alpha_convert_matrix_wapper(common_arg->format, common_arg->alpha_descr, common_arg->layout, 
                                cooA, &compute_matrix, param0, param1);
  }
  alpha_timer_t timer;
  // double total_time = 0.;
  double onetime = 0.;
  std::vector<double> times;
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    if (common_arg->warm)
    for (int i = 0; i < 20; i++){
      alpha_call_exit(
          alphasparse_s_mv(common_arg->transA, *((float *)alpha_alpha_char),
                           compute_matrix, common_arg->alpha_descr,
                           (float *)x_char, *((float *)alpha_beta_char),
                           (float *)icty_char),
          "alphasparse_s_mv");
    }
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_timing_start(&timer);
      alpha_call_exit(
          alphasparse_s_mv(common_arg->transA, *((float *)alpha_alpha_char),
                           compute_matrix, common_arg->alpha_descr,
                           (float *)x_char, *((float *)alpha_beta_char),
                           (float *)icty_char),
          "alphasparse_s_mv");
      alpha_timing_end(&timer);
      onetime = alpha_timing_elapsed_time(&timer);
      times.push_back(onetime);
    }

  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    if (common_arg->warm)
    for (int i = 0; i < 20; i++){
      alpha_call_exit(
          alphasparse_d_mv(common_arg->transA, *((double *)alpha_alpha_char),
                           compute_matrix, common_arg->alpha_descr,
                           (double *)x_char, *((double *)alpha_beta_char),
                           (double *)icty_char),
          "alphasparse_s_mv");
    }
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_timing_start(&timer);
          alphasparse_d_mv(common_arg->transA, *((double *)alpha_alpha_char),
                           compute_matrix, common_arg->alpha_descr,
                           (double *)x_char, *((double *)alpha_beta_char),
                           (double *)icty_char);
      alpha_timing_end(&timer);
      onetime = alpha_timing_elapsed_time(&timer);
      times.push_back(onetime);
    }

  // } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
  //   if (common_arg->warm)
  //     alpha_call_exit(alphasparse_c_mv(common_arg->transA,
  //                                      *((ALPHA_Complex8 *)alpha_alpha_char),
  //                                      compute_matrix, common_arg->alpha_descr,
  //                                      (ALPHA_Complex8 *)x_char,
  //                                      *((ALPHA_Complex8 *)alpha_beta_char),
  //                                      (ALPHA_Complex8 *)icty_char),
  //                     "alphasparse_c_mv");
  //   alpha_timing_start(&timer);

  //   for (int i = 0; i < common_arg->iter; i++) {
  //     alpha_call_exit(alphasparse_c_mv(common_arg->transA,
  //                                      *((ALPHA_Complex8 *)alpha_alpha_char),
  //                                      compute_matrix, common_arg->alpha_descr,
  //                                      (ALPHA_Complex8 *)x_char,
  //                                      *((ALPHA_Complex8 *)alpha_beta_char),
  //                                      (ALPHA_Complex8 *)icty_char),
  //                     "alphasparse_c_mv");
  //   }
  //   alpha_timing_end(&timer);
  //   total_time = alpha_timing_elapsed_time(&timer);
  // } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
  //   if (common_arg->warm)
  //     alpha_call_exit(alphasparse_z_mv(common_arg->transA,
  //                                      *((ALPHA_Complex16 *)alpha_alpha_char),
  //                                      compute_matrix, common_arg->alpha_descr,
  //                                      (ALPHA_Complex16 *)x_char,
  //                                      *((ALPHA_Complex16 *)alpha_beta_char),
  //                                      (ALPHA_Complex16 *)icty_char),
  //                     "alphasparse_c_mv");
  //   alpha_timing_start(&timer);
  //   for (int i = 0; i < common_arg->iter; i++) {
  //     alpha_call_exit(alphasparse_z_mv(common_arg->transA,
  //                                      *((ALPHA_Complex16 *)alpha_alpha_char),
  //                                      compute_matrix, common_arg->alpha_descr,
  //                                      (ALPHA_Complex16 *)x_char,
  //                                      *((ALPHA_Complex16 *)alpha_beta_char),
  //                                      (ALPHA_Complex16 *)icty_char),
  //                     "alphasparse_c_mv");
  //   }
  //   alpha_timing_end(&timer);
  //   total_time = alpha_timing_elapsed_time(&timer);
  }
  char str[20];
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO){
    strcpy(str, "coo");
  }
  else if (common_arg->format == ALPHA_SPARSE_FORMAT_CSR){
    strcpy(str, "csr");
  }
  else if (common_arg->format == ALPHA_SPARSE_FORMAT_DIA){
    strcpy(str, "dia");
  }
  else if (common_arg->format == ALPHA_SPARSE_FORMAT_ELL){
    strcpy(str, "ell");
  }
  else if (common_arg->format == ALPHA_SPARSE_FORMAT_BSR){
    strcpy(str, "bsr");
  }
  else if (common_arg->format == ALPHA_SPARSE_FORMAT_SELL_C_SIGMA){
    sprintf(str, "sell-%d-%d", param0, param1);
  }
  printf("%s_time : %lf\n", str,
         get_avg_time(times) * 1000);

  alphasparse_destroy(cooA);
  if (common_arg->format != ALPHA_SPARSE_FORMAT_COO)
    alphasparse_destroy(compute_matrix);
}
static void alpha_mv_plain(matrix_data_t *matrix_data,
                           alpha_common_args_t *common_arg, const char *x_char,
                           char *icty_char, const char *alpha_alpha_char,
                           const char *alpha_beta_char) {
  alphasparse_matrix_t cooA, compute_matrix;
  alpha_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix = cooA;
  } else {
    alpha_convert_matrix_wapper(ALPHA_SPARSE_FORMAT_CSR,
                                common_arg->alpha_descr, cooA, &compute_matrix,
                                param0, param1);
  }

  alpha_timer_t timer;
  double total_time = 0.;
  alpha_set_thread_num(1);
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    if (common_arg->warm)
      alpha_call_exit(
          alphasparse_s_mv_plain(
              common_arg->transA, *((float *)alpha_alpha_char), compute_matrix,
              common_arg->alpha_descr, (float *)x_char,
              *((float *)alpha_beta_char), (float *)icty_char),
          "alphasparse_s_mv_plain");
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(
          alphasparse_s_mv_plain(
              common_arg->transA, *((float *)alpha_alpha_char), compute_matrix,
              common_arg->alpha_descr, (float *)x_char,
              *((float *)alpha_beta_char), (float *)icty_char),
          "alphasparse_s_mv_plain");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    if (common_arg->warm)
      alpha_call_exit(
          alphasparse_d_mv_plain(
              common_arg->transA, *((double *)alpha_alpha_char), compute_matrix,
              common_arg->alpha_descr, (double *)x_char,
              *((double *)alpha_beta_char), (double *)icty_char),
          "alphasparse_d_mv_plain");
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(
          alphasparse_d_mv_plain(
              common_arg->transA, *((double *)alpha_alpha_char), compute_matrix,
              common_arg->alpha_descr, (double *)x_char,
              *((double *)alpha_beta_char), (double *)icty_char),
          "alphasparse_d_mv_plain");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    if (common_arg->warm)
      alpha_call_exit(
          alphasparse_c_mv_plain(
              common_arg->transA, *((ALPHA_Complex8 *)alpha_alpha_char),
              compute_matrix, common_arg->alpha_descr, (ALPHA_Complex8 *)x_char,
              *((ALPHA_Complex8 *)alpha_beta_char),
              (ALPHA_Complex8 *)icty_char),
          "alphasparse_c_mv_plain");
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(
          alphasparse_c_mv_plain(
              common_arg->transA, *((ALPHA_Complex8 *)alpha_alpha_char),
              compute_matrix, common_arg->alpha_descr, (ALPHA_Complex8 *)x_char,
              *((ALPHA_Complex8 *)alpha_beta_char),
              (ALPHA_Complex8 *)icty_char),
          "alphasparse_c_mv_plain");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    if (common_arg->warm)
      alpha_call_exit(
          alphasparse_z_mv_plain(
              common_arg->transA, *((ALPHA_Complex16 *)alpha_alpha_char),
              compute_matrix, common_arg->alpha_descr,
              (ALPHA_Complex16 *)x_char, *((ALPHA_Complex16 *)alpha_beta_char),
              (ALPHA_Complex16 *)icty_char),
          "alphasparse_z_mv_plain");
    alpha_timing_start(&timer);
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_call_exit(
          alphasparse_z_mv_plain(
              common_arg->transA, *((ALPHA_Complex16 *)alpha_alpha_char),
              compute_matrix, common_arg->alpha_descr,
              (ALPHA_Complex16 *)x_char, *((ALPHA_Complex16 *)alpha_beta_char),
              (ALPHA_Complex16 *)icty_char),
          "alphasparse_z_mv_plain");
    }
    alpha_timing_end(&timer);
    total_time = alpha_timing_elapsed_time(&timer);
  }
  printf("%s time : %lf[ms]\n", "alphasparse_mv_plain",
         (total_time / common_arg->iter) * 1000);

  alphasparse_destroy(cooA);
  if (common_arg->format != ALPHA_SPARSE_FORMAT_COO)
    alphasparse_destroy(compute_matrix);
}
#ifdef __MKL__
static void mkl_mv(matrix_data_t *matrix_data, alpha_common_args_t *common_arg,
                   const char *x_char, char *icty_char,
                   const char *alpha_alpha_char, const char *alpha_beta_char) {
  sparse_matrix_t cooA, compute_matrix;
  mkl_create_coo_wapper(matrix_data, common_arg->data_type, &cooA);
  if (common_arg->format == ALPHA_SPARSE_FORMAT_COO) {
    compute_matrix = cooA;
  } else {
    // mkl_convert_matrix_wapper(ALPHA_SPARSE_FORMAT_CSR, common_arg->mkl_descr, common_arg->mkl_layout, 
    //                           cooA, &compute_matrix, param0, param1);
    mkl_convert_matrix_wapper(common_arg->format, common_arg->mkl_descr, common_arg->mkl_layout, 
                              cooA, &compute_matrix, param0, param1);
  }
  struct matrix_descr descr;
  // descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  // mkl_sparse_set_mv_hint(compute_matrix, SPARSE_OPERATION_NON_TRANSPOSE, descr, 100);
  // mkl_sparse_optimize(compute_matrix);
  // 执行优化预处理
  alpha_timer_t timer;
  double onetime = 0.;
  std::vector<double> times;
  mkl_set_num_threads(common_arg->thread_num);
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    if (common_arg->warm)
      for (int i = 0; i < 20; i++){
      mkl_call_exit(mkl_sparse_s_mv(
                        common_arg->mkl_transA, *((float *)alpha_alpha_char),
                        compute_matrix, common_arg->mkl_descr, (float *)x_char,
                        *((float *)alpha_beta_char), (float *)icty_char),
                    "mkl_sparse_s_mv");
      }

    
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_timing_start(&timer);
      mkl_call_exit(mkl_sparse_s_mv(
                        common_arg->mkl_transA, *((float *)alpha_alpha_char),
                        compute_matrix, common_arg->mkl_descr, (float *)x_char,
                        *((float *)alpha_beta_char), (float *)icty_char),
                    "mkl_sparse_s_mv");
      alpha_timing_end(&timer);
      onetime = alpha_timing_elapsed_time(&timer);
      times.push_back(onetime);
    }
    
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    if (common_arg->warm)
    for (int i = 0; i < 20; i++){
      mkl_call_exit(mkl_sparse_d_mv(
                        common_arg->mkl_transA, *((double *)alpha_alpha_char),
                        compute_matrix, common_arg->mkl_descr, (double *)x_char,
                        *((double *)alpha_beta_char), (double *)icty_char),
                    "mkl_sparse_d_mv");
    }
    
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_timing_start(&timer);
      mkl_call_exit(mkl_sparse_d_mv(
                        common_arg->mkl_transA, *((double *)alpha_alpha_char),
                        compute_matrix, common_arg->mkl_descr, (double *)x_char,
                        *((double *)alpha_beta_char), (double *)icty_char),
                    "mkl_sparse_d_mv");
      alpha_timing_end(&timer);
      onetime = alpha_timing_elapsed_time(&timer);
      times.push_back(onetime);
    }
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    if (common_arg->warm)
      mkl_call_exit(
          mkl_sparse_c_mv(
              common_arg->mkl_transA, *((MKL_Complex8 *)alpha_alpha_char),
              compute_matrix, common_arg->mkl_descr, (MKL_Complex8 *)x_char,
              *((MKL_Complex8 *)alpha_beta_char), (MKL_Complex8 *)icty_char),
          "mkl_sparse_c_mv");
    
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_timing_start(&timer);
      mkl_call_exit(
          mkl_sparse_c_mv(
              common_arg->mkl_transA, *((MKL_Complex8 *)alpha_alpha_char),
              compute_matrix, common_arg->mkl_descr, (MKL_Complex8 *)x_char,
              *((MKL_Complex8 *)alpha_beta_char), (MKL_Complex8 *)icty_char),
          "mkl_sparse_c_mv");
      alpha_timing_end(&timer);
      onetime = alpha_timing_elapsed_time(&timer);
      times.push_back(onetime);
    }
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    if (common_arg->warm)
      mkl_call_exit(
          mkl_sparse_z_mv(
              common_arg->mkl_transA, *((MKL_Complex16 *)alpha_alpha_char),
              compute_matrix, common_arg->mkl_descr, (MKL_Complex16 *)x_char,
              *((MKL_Complex16 *)alpha_beta_char), (MKL_Complex16 *)icty_char),
          "mkl_sparse_z_mv");
    for (int i = 0; i < common_arg->iter; i++) {
      alpha_timing_start(&timer);
      mkl_call_exit(
          mkl_sparse_z_mv(
              common_arg->mkl_transA, *((MKL_Complex16 *)alpha_alpha_char),
              compute_matrix, common_arg->mkl_descr, (MKL_Complex16 *)x_char,
              *((MKL_Complex16 *)alpha_beta_char), (MKL_Complex16 *)icty_char),
          "mkl_sparse_z_mv");
      alpha_timing_end(&timer);
      onetime = alpha_timing_elapsed_time(&timer);
      times.push_back(onetime);    
      }

  }
    char str[20];
  if(common_arg->format == ALPHA_SPARSE_FORMAT_COO){
    strcpy(str, "mkl_coo_time");
  }else if(common_arg->format == ALPHA_SPARSE_FORMAT_BSR){
    strcpy(str, "mkl_bsr_time");
  }else if(common_arg->format == ALPHA_SPARSE_FORMAT_CSR){
    strcpy(str, "mkl_csr_time");
  }
  printf("%s : %lf\n", str,
         get_avg_time(times) * 1000);

  mkl_sparse_destroy(cooA);
  if (common_arg->format != ALPHA_SPARSE_FORMAT_COO)
    mkl_sparse_destroy(compute_matrix);
}
#endif

int main(int argc, const char *argv[]) {
  // args
  alpha_common_args_t common_arg;
  // read_coo
  matrix_data_t matrix_data;
  args_help(argc, argv);
  // init args
  parse_args_and_initialize(argc, argv, &common_arg);
  param0 = common_arg.param0;
  param1 = common_arg.param1;
  // printf("C %d sigma %d\n", param0, param1);
  alpha_timer_t timer;
  alpha_timing_start(&timer);
  alpha_read_coo_wrapper(&matrix_data, &common_arg, FILE_SOURCE, param0);
  alpha_timing_end(&timer);
  double time_elapsed = alpha_timing_elapsed_time(&timer);
  // printf("io elapesd %f [ms]\n", time_elapsed * 1e3);
  char *x_char;
  char *icty_char;
  char *icty_plain_char;
  char *alpha_alpha_char;
  char *alpha_beta_char;

#ifdef __MKL__
  char *mkl_alpha_char;
  char *mkl_beta_char;
#endif

  int rows, cols;

  alpha_alpha_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);
  alpha_beta_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);

#ifdef __MKL__
  mkl_alpha_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);
  mkl_beta_char = (char *)alpha_malloc(bytes_type[common_arg.data_type]);
#endif
  alpha_initialize_alpha_beta(alpha_alpha_char, alpha_beta_char,
                              common_arg.data_type);
#ifdef __MKL__
  mkl_initialize_alpha_beta(mkl_alpha_char, mkl_beta_char,
                            common_arg.data_type);
#endif
  if (common_arg.transA == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    rows = matrix_data.k;
    cols = matrix_data.m;
  } else {
    rows = matrix_data.m;
    cols = matrix_data.k;
  }

  malloc_random_fill(common_arg.data_type, (void **)&x_char, cols, 0);  // x,y
  malloc_random_fill(common_arg.data_type, (void **)&icty_char, rows,
                     1);  // x,y
  malloc_random_fill(common_arg.data_type, (void **)&icty_plain_char, rows,
                     1);  // x,y

  common_arg.thread_num = 1;
  // alpha_mv_autoSelect(&matrix_data, &common_arg, x_char, icty_char, alpha_alpha_char,
  //           alpha_beta_char);
  alpha_mv(&matrix_data, &common_arg, x_char, icty_char, alpha_alpha_char,
           alpha_beta_char);

  // mkl_mv_dia(&matrix_data, &common_arg, x_char, icty_char, alpha_alpha_char,
  //          alpha_beta_char);
  
  if (common_arg.check) {

#ifdef __MKL__
    mkl_mv(&matrix_data, &common_arg, x_char, icty_plain_char, mkl_alpha_char,
           mkl_beta_char);

// #else
//     alpha_mv_plain(&matrix_data, &common_arg, x_char, icty_plain_char,
//                    alpha_alpha_char, alpha_beta_char);
#endif
    check_arm(common_arg.data_type, icty_char, rows, icty_plain_char);
    if (common_arg.data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
      ALPHA_Complex16 *icty = (ALPHA_Complex16 *)icty_char;
      ALPHA_Complex16 *ref = (ALPHA_Complex16 *)icty_plain_char;

      for (int i = 0; i < 20; i++) {
        printf("y[%d] (%-4.8f,%-4.8f) vs  (%-4.8f,%-4.8f)\n", i,
        icty[i].real,
               icty[i].imag, ref[i].real, ref[i].imag);
      }
    }
    if (common_arg.data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
      ALPHA_Complex8 *icty = (ALPHA_Complex8 *)icty_char;
      ALPHA_Complex8 *ref = (ALPHA_Complex8 *)icty_plain_char;

      for (int i = 0; i < 20; i++) {
        printf("[%d] (%-4.8f,%-4.8f) vs  (%-4.8f,%-4.8f)\n", i, icty[i].real,
               icty[i].imag, ref[i].real, ref[i].imag);
      }
    }
    alpha_free(icty_plain_char);
  }

  alpha_free(x_char);
  alpha_free(icty_char);
  destory_matrix_data(&matrix_data);
  return 0;
}
