#pragma once

/**
 * @brief header for all utils
 */

#ifndef index2
#define index2(y, x, ldx) ((x) + (ldx) * (y))
#endif // !index2

#ifndef index3
#define index3(z, y, x, ldy, ldx) index2(index2(z, y, ldy), x, ldx)
#endif // !index3

#ifndef index4
#define index4(d, c, b, a, ldc, ldb, lda) index2(index2(index2(d, c, ldc), b, ldb), a, lda)
#endif // !index4

#ifndef alpha_min
#define alpha_min(x, y) ((x) < (y) ? (x) : (y))
#endif // !alpha_min

#ifndef alpha_max
#define alpha_max(x, y) ((x) < (y) ? (y) : (x))
#endif // !alpha_max

#define CEIL(x,y) (((x)+((y)-1))/(y))

#ifdef __CUDA__
#define GPU_TIMER_START(elapsed_time, event_start, event_stop) \
  do                                                           \
  {                                                            \
    elapsed_time = 0.0;                                        \
    cudaEventCreate(&event_start);                             \
    cudaEventCreate(&event_stop);                              \
    cudaEventRecord(event_start);                              \
  } while (0)

#define GPU_TIMER_END(elapsed_time, event_start, event_stop)      \
  do                                                              \
  {                                                               \
    cudaEventRecord(event_stop);                                  \
    cudaEventSynchronize(event_stop);                             \
    cudaEventElapsedTime(&elapsed_time, event_start, event_stop); \
  } while (0)

#define CHECK_CUDA(func)                                         \
  {                                                              \
    cudaError_t status = (func);                                 \
    if (status != cudaSuccess)                                   \
    {                                                            \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                           \
             cudaGetErrorString(status),                         \
             status);                                            \
      exit(-1);                                                  \
    }                                                            \
  }

#define CHECK_CUSPARSE(func)                                         \
  {                                                                  \
    cusparseStatus_t status = (func);                                \
    if (status != CUSPARSE_STATUS_SUCCESS)                           \
    {                                                                \
      printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                               \
             cusparseGetErrorString(status),                         \
             status);                                                \
      exit(-1);                                                      \
    }                                                                \
  }
#endif
