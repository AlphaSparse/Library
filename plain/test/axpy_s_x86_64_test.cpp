#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"

static void mkl_axpyi(const MKL_INT n, const float a, const float *x,
                      const MKL_INT *indx, float *y, int thread_num) {
  mkl_set_num_threads(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  cblas_saxpyi(n, a, x, indx, y);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "cblas_saxpyi");
}

static void alpha_axpyi(const ALPHA_INT n, const float a, const float *x,
                      const ALPHA_INT *indx, float *y, int thread_num) {
  alpha_set_thread_num(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_s_axpy_plain(n, a, x, indx, y), "alpha_s_axpyi");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alpha_s_axpyi");
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  int thread_num = args_get_thread_num(argc, argv);
  bool check = args_get_if_check(argc, argv);

  const float alpha = 3.;

  // return
  MKL_INT mkl_n = 50;
  float a = 2.0f;
  ALPHA_INT alpha_n = 50;

  ALPHA_INT *alpha_incx = (ALPHA_INT*)alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
  MKL_INT *mkl_incx = (MKL_INT*)alpha_memalign(sizeof(MKL_INT) * alpha_n, DEFAULT_ALIGNMENT);

  float *x = (float*)alpha_memalign(sizeof(float) * alpha_n, DEFAULT_ALIGNMENT);
  alpha_fill_random_s(x, 1, alpha_n);

  for (ALPHA_INT i = 0; i < alpha_n; i++) {
    mkl_incx[i] = i * 20;
    alpha_incx[i] = i * 20;
  }

  float *mkl_y = (float*)alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);
  float *alpha_y = (float*)alpha_memalign(sizeof(float) * alpha_n * 20, DEFAULT_ALIGNMENT);

  alpha_fill_random_s(mkl_y, 1, alpha_n * 20);
  alpha_fill_random_s(alpha_y, 1, alpha_n * 20);

  alpha_axpyi(alpha_n, a, x, alpha_incx, alpha_y, thread_num);
  int status = 0;
  if (check) {
    mkl_axpyi(mkl_n, a, x, mkl_incx, mkl_y, thread_num);
    status = check_s(mkl_y, mkl_n * 20, alpha_y, alpha_n * 20);
  }

  return status;
}