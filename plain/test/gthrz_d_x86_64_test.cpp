#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"

static void mkl_gthrz(const MKL_INT n, double *x, const MKL_INT *indx,
                      double *y, int thread_num) {
  mkl_set_num_threads(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  cblas_dgthrz(n, y, x, indx);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "cblas_sgthrz");
}

static void alpha_gthrz(const ALPHA_INT n, double *x, const ALPHA_INT *indx,
                      double *y, int thread_num) {
  alpha_set_thread_num(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_d_gthrz_plain(n, y, x, indx), "alpha_d_gthrz");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alpha_d_gthrz");
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  int thread_num = args_get_thread_num(argc, argv);
  bool check = args_get_if_check(argc, argv);

  const double alpha = 3.;

  // return
  MKL_INT mkl_n = 50;
  double a = 2.0f;
  ALPHA_INT alpha_n = 50;

  ALPHA_INT *alpha_incx = (ALPHA_INT*)alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
  MKL_INT *mkl_incx = (MKL_INT*)alpha_memalign(sizeof(MKL_INT) * mkl_n, DEFAULT_ALIGNMENT);

  double *alpha_x = (double*)alpha_memalign(sizeof(double) * alpha_n, DEFAULT_ALIGNMENT);
  double *mkl_x = (double*)alpha_memalign(sizeof(double) * mkl_n, DEFAULT_ALIGNMENT);
  // alpha_fill_random_s(x, 1, alpha_n);

  for (ALPHA_INT i = 0; i < alpha_n; i++) {
    mkl_incx[i] = i * 20;
    alpha_incx[i] = i * 20;
  }

  double *alpha_y = (double*)alpha_memalign(sizeof(double) * alpha_n * 20, DEFAULT_ALIGNMENT);
  double *mkl_y = (double*)alpha_memalign(sizeof(double) * mkl_n * 20, DEFAULT_ALIGNMENT);

  alpha_fill_random_d(alpha_y, 1, alpha_n * 20);
  alpha_fill_random_d(mkl_y, 1, mkl_n * 20);

  alpha_gthrz(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
  int status = 0;
  if (check) {
    mkl_gthrz(mkl_n, mkl_x, mkl_incx, mkl_y, thread_num);
    status = check_d(mkl_x, mkl_n, alpha_x, alpha_n);
  }

  return status;
}