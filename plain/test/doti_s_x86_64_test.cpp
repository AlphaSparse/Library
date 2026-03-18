#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"

float mkl_doti(const MKL_INT n, const float *x, const MKL_INT *indx,
               const float *y, int thread_num) {
  mkl_set_num_threads(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  float res = cblas_sdoti(n, x, indx, y);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "cblas_sdoti");

  return res;
}

float alpha_doti(const ALPHA_INT n, const float *x, const ALPHA_INT *indx,
               const float *y, int thread_num) {
  alpha_set_thread_num(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  float res = alphasparse_s_doti_plain(n, x, indx, y);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alpha_s_doti");

  return res;
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  int thread_num = args_get_thread_num(argc, argv);
  bool check = args_get_if_check(argc, argv);

  // return
  MKL_INT mkl_n = 50;
  ALPHA_INT alpha_n = 50;

  ALPHA_INT *alpha_incx = (ALPHA_INT *)(ALPHA_INT*)alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
  MKL_INT *mkl_incx = (MKL_INT *)(MKL_INT*)alpha_memalign(sizeof(MKL_INT) * alpha_n, DEFAULT_ALIGNMENT);

  float *alpha_x = (float *)(float*)alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
  float *mkl_x = (float *)(float*)alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
  // alpha_fill_random_s(x, 1, alpha_n);

  for (ALPHA_INT i = 0; i < alpha_n; i++) {
    mkl_incx[i] = i * 20;
    alpha_incx[i] = i * 20;
  }

  float *alpha_y = (float *)(float*)alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);

  alpha_fill_random_s(alpha_y, 1, alpha_n * 20);

  float alpha_res = alpha_doti(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
  int status = 0;
  if (check) {
    float mkl_res = mkl_doti(mkl_n, mkl_x, mkl_incx, alpha_y, thread_num);
    status = alpha_res == mkl_res;
    if (status) {
      fprintf(stderr, "doti_s correct\n");
    } else {
      fprintf(stderr, "doti_s error\n");
    }
  }

  return status;
}