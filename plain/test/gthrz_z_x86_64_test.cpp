#include <alphasparse.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"

static void mkl_gthrz(const MKL_INT n, MKL_Complex16 *x, const MKL_INT *indx,
                      MKL_Complex16 *y, int thread_num) {
  mkl_set_num_threads(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  cblas_zgthrz(n, y, x, indx);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "cblas_sgthrz");
}

static void alpha_gthrz(const ALPHA_INT n, ALPHA_Complex16 *x, const ALPHA_INT *indx,
                      ALPHA_Complex16 *y, int thread_num) {
  alpha_set_thread_num(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alpha_call_exit(alphasparse_z_gthrz_plain(n, y, x, indx), "alpha_z_gthrz");

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alpha_z_gthrz");
}

int main(int argc, const char *argv[]) {
  // args
  args_help(argc, argv);
  int thread_num = args_get_thread_num(argc, argv);
  bool check = args_get_if_check(argc, argv);

  // return
  MKL_INT mkl_n = 50;
  ALPHA_INT alpha_n = 50;

  ALPHA_INT *alpha_incx = (ALPHA_INT*)alpha_memalign(sizeof(ALPHA_INT) * alpha_n, DEFAULT_ALIGNMENT);
  MKL_INT *mkl_incx = (MKL_INT*)alpha_memalign(sizeof(MKL_INT) * mkl_n, DEFAULT_ALIGNMENT);

  ALPHA_Complex16 *alpha_x =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * alpha_n, DEFAULT_ALIGNMENT);
  MKL_Complex16 *mkl_x =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * mkl_n, DEFAULT_ALIGNMENT);
  // alpha_fill_random_s(x, 1, alpha_n);

  for (ALPHA_INT i = 0; i < alpha_n; i++) {
    mkl_incx[i] = i * 20;
    alpha_incx[i] = i * 20;
  }

  ALPHA_Complex16 *alpha_y =
      (ALPHA_Complex16*)alpha_memalign(sizeof(ALPHA_Complex16) * alpha_n * 20, DEFAULT_ALIGNMENT);
  MKL_Complex16 *mkl_y =
      (MKL_Complex16*)alpha_memalign(sizeof(MKL_Complex16) * mkl_n * 20, DEFAULT_ALIGNMENT);

  alpha_fill_random_z(alpha_y, 1, alpha_n * 20);
  alpha_fill_random_d((double *)mkl_y, 1, 2 * mkl_n * 20);

  alpha_gthrz(alpha_n, alpha_x, alpha_incx, alpha_y, thread_num);
  int status = 0;
  if (check) {
    mkl_gthrz(mkl_n, mkl_x, mkl_incx, mkl_y, thread_num);
    status = check_d((double *)mkl_x, 2 * mkl_n, (double *)alpha_x, 2 * alpha_n);
  }

  return status;
}