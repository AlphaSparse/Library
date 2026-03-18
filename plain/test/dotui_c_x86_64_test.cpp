#include <alphasparse.h>
#include <memory.h>
#include <mkl.h>
#include <stdio.h>

#include "test_common.h"

static void mkl_dotui(const MKL_INT n, const MKL_Complex8 *x,
                      const MKL_INT *indx, const MKL_Complex8 *y,
                      MKL_Complex8 *dotui, int thread_num) {
  mkl_set_num_threads(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  cblas_cdotui_sub(n, (void *)x, indx, (void *)y, (void *)dotui);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "cblas_cdotui");

  return;
}

static void alpha_dotui(const ALPHA_INT n, const ALPHA_Complex8 *x,
                      const ALPHA_INT *indx, const ALPHA_Complex8 *y,
                      ALPHA_Complex8 *dotui, int thread_num) {
  alpha_set_thread_num(thread_num);
  alpha_timer_t timer;
  alpha_timing_start(&timer);

  alphasparse_c_dotui_sub_plain(n, x, indx, y, dotui);

  alpha_timing_end(&timer);
  alpha_timing_elaped_time_print(&timer, "alpha_s_dotui");

  return;
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
  MKL_INT *mkl_incx = (MKL_INT*)alpha_memalign(sizeof(MKL_INT) * alpha_n, DEFAULT_ALIGNMENT);

  ALPHA_Complex8 *alpha_x =
      (ALPHA_Complex8*)alpha_memalign(sizeof(ALPHA_Complex8) * alpha_n, DEFAULT_ALIGNMENT);
  MKL_Complex8 *mkl_x =
      (MKL_Complex8*)(MKL_Complex8*)alpha_memalign(sizeof(MKL_Complex8) * alpha_n, DEFAULT_ALIGNMENT);
  // alpha_fill_random_s(x, 1, alpha_n);

  for (ALPHA_INT i = 0; i < alpha_n; i++) {
    mkl_incx[i] = i * 20;
    alpha_incx[i] = i * 20;
  }

  ALPHA_Complex8 *alpha_y =
      (ALPHA_Complex8*)alpha_memalign(sizeof(ALPHA_Complex8) * alpha_n * 20, DEFAULT_ALIGNMENT);
  MKL_Complex8 *mkl_y =
      (MKL_Complex8*)alpha_memalign(sizeof(MKL_Complex8) * mkl_n * 20, DEFAULT_ALIGNMENT);

  alpha_fill_random_c(alpha_y, 1, alpha_n * 20);
  alpha_fill_random_s((float *)mkl_y, 1, 2 * alpha_n * 20);

  MKL_Complex8 *mkl_dotui_v =
      (MKL_Complex8*)alpha_memalign(sizeof(MKL_Complex8), DEFAULT_ALIGNMENT);
  ALPHA_Complex8 *alpha_dotui_v =
      (ALPHA_Complex8*)alpha_memalign(sizeof(ALPHA_Complex8), DEFAULT_ALIGNMENT);

  memset(mkl_dotui_v, 0, sizeof(MKL_Complex8));
  memset(alpha_dotui_v, 0, sizeof(ALPHA_Complex8));

  alpha_dotui(alpha_n, alpha_x, alpha_incx, alpha_y, alpha_dotui_v, thread_num);
  int status = 0;
  if (check) {
    mkl_dotui(mkl_n, mkl_x, mkl_incx, mkl_y, mkl_dotui_v, thread_num);
    status = check_s((float *)alpha_dotui_v, 2, (float *)mkl_dotui_v, 2);
  }

  return status;
}