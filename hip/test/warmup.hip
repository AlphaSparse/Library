#include "include/args.h"
#include "include/io.h"
#include "../format/coo_order.h"
#include "../test/include/args.h"
#include "../format/coo2csr.h"
#include "alphasparse/util/error.h"

#define BLOCK_M 16
#define BLOCK_N 16

__global__ void warmup_gemm(double *a, double *b, double *c, int m, int n, int k) {
  int tid = threadIdx.x;
  int rowid = blockIdx.x * BLOCK_M + tid / BLOCK_N;
  int colid = blockIdx.y * BLOCK_N + tid % BLOCK_N;
  double acc = 0.0;
  for (int i = 0; i < k; ++i) {
    acc += a[rowid * k + i] * b[i * n + colid];
  }

  c[rowid * n + colid] = acc;
}

void warm_func()
{
  int m = 8192;
  int n = 8192;
  int k = 256;
  int a_size = m * k;
  int b_size = n * k;
  int c_size = m * n;
  double *h_a = (double*)alpha_malloc(a_size * sizeof(double));
  double *h_b = (double*)alpha_malloc(b_size * sizeof(double));
  double *h_c = (double*)alpha_malloc(c_size * sizeof(double));
  alpha_fill_random(h_a, 0, a_size);
  alpha_fill_random(h_b, 1, b_size);

  double *d_a, *d_b, *d_c;
  hipMalloc(&d_a, a_size * sizeof(double));
  hipMalloc(&d_b, b_size * sizeof(double));
  hipMalloc(&d_c, c_size * sizeof(double));
  hipMemcpy(d_a, h_a, a_size * sizeof(double), hipMemcpyHostToDevice);
  hipMemcpy(d_b, h_b, b_size * sizeof(double), hipMemcpyHostToDevice);

  hipEvent_t event_start, event_stop;
  float elapsed_time = 0.f;
  float time_warm = 0;
  dim3 dg(m / BLOCK_M, n / BLOCK_N, 1);
  dim3 db(256, 1, 1);

  while (time_warm < 500.0)
  {
    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    warmup_gemm<<<dg, db>>>(d_a, d_b, d_c, m, n, k);
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    time_warm = time_warm + elapsed_time;
  }
}