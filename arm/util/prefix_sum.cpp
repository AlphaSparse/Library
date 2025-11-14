#include "alphasparse/util/prefix_sum.h"

#include <stdio.h>
#include <string.h>

#include "alphasparse/util/malloc.h"
#include "alphasparse/util/thread.h"
#ifdef __aarch64__
#include <arm_neon.h>
#endif
static void vector_add(int* inout, int increment, size_t len) {
  size_t i = 0;
#ifdef __aarch64__
  int32x4_t v0, v1, v2, v3;
  int32x4_t v_inc = vdupq_n_s32(increment);
  for (; i + 15 < len; i += 16) {
    v0 = vld1q_s32((inout + i));
    v1 = vld1q_s32((inout + i + 4));
    v2 = vld1q_s32((inout + i + 8));
    v3 = vld1q_s32((inout + i + 12));

    v0 = vaddq_s32(v_inc, v0);
    v1 = vaddq_s32(v_inc, v1);
    v2 = vaddq_s32(v_inc, v2);
    v3 = vaddq_s32(v_inc, v3);

    vst1q_s32(inout + i, v0);
    vst1q_s32(inout + i + 4, v1);
    vst1q_s32(inout + i + 8, v2);
    vst1q_s32(inout + i + 12, v3);
  }
#endif
  for (; i < len; i++) {
    inout[i] += increment;
  }
}
inline void prefix_sum_single_thread(prefix_sum_type_t scan_type, const int32_t* source,
                                     const size_t len, int32_t* output)

{
  if (len <= 0) {
    printf("input array is empty\n");
    exit(-1);
  }
  if (scan_type == INC_SCAN) {
    output[0] = source[0];
    for (size_t i = 1; i < len; i++) {
      output[i] = source[i] + output[i - 1];
    }
  }

  else {
    // in case in-place
    int pre = source[0];
    int cur = 0;
    output[0] = 0;
    for (size_t i = 1; i < len; i++) {
      cur = source[i];
      output[i] = pre + output[i - 1];
      pre = cur;
    }
  }
}
void prefix_sum(prefix_sum_type_t scan_type, const int32_t* source, const size_t len,
                int32_t* output) {
#ifdef _OPENMP
  int thread_num = alpha_get_thread_num();
#else
  int thread_num = 1;
#endif
  int* temp_buffer = (int*)alpha_malloc(sizeof(int32_t) * thread_num);
  const int len_per_thread = len / thread_num;

  memset(temp_buffer, 0, (sizeof(int32_t) * thread_num));

#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    const int tid = alpha_get_thread_id();
    const int start = len * tid / thread_num;
    const int end = len * (tid + 1) / thread_num;
    if (end > start) {
      if (scan_type == EXL_SCAN) {
        temp_buffer[tid] = source[end - 1];
      } else if (scan_type == INC_SCAN) {
        temp_buffer[tid] = 0;
      }
      prefix_sum_single_thread(scan_type, source + start, end - start, output + start);
      temp_buffer[tid] += output[end - 1];
    } else {
      temp_buffer[tid] = 0;
    }
  }

  prefix_sum_single_thread(EXL_SCAN, temp_buffer, thread_num, temp_buffer);

#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    const int tid = alpha_get_thread_id();
    const int start = len * tid / thread_num;
    const int end = len * (tid + 1) / thread_num;
    vector_add(output + start, temp_buffer[tid], end - start);
  }

  alpha_free(temp_buffer);
}