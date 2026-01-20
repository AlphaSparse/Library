/**
 * @brief implement for malloc utils
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/util/malloc.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "alphasparse/util/random.h"
#include "alphasparse/util/thread.h"

#ifdef NUMA
#include <numa.h>
#endif

void *alpha_malloc(size_t bytes) {
#ifdef NUMA
  void *ret = numa_alloc_onnode(bytes, 0);
#else
  void *ret = malloc(bytes);
#endif
  if (ret == NULL) {
    printf("no enough memory space to alloc!!!\n");
    exit(-1);
  }
  return ret;
}

void *alpha_memalign(size_t bytes, size_t alignment) {
#ifdef NUMA
  void *ret = numa_alloc_onnode(bytes, 0);
#else
  void *ret = memalign(alignment, bytes);
#endif
  if (ret == NULL) {
    printf("no enough memory space to alloc!!!");
    exit(-1);
  }
  return ret;
}

void alpha_free(void *point) { 
  if (!point)
    free(point); 
}

void alpha_clear_cache() {
  ALPHA_INT thread_num = alpha_get_thread_num();
  const size_t L3_used = (thread_num + 23) / 24;
  const size_t size = L3_CACHE_SIZE * 8 * L3_used;
  long long *c = (long long *)alpha_memalign(size, DEFAULT_ALIGNMENT);
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < 15; i++)
    for (size_t j = 0; j < L3_CACHE_SIZE * L3_used; j++) c[j] += i * j;
  alpha_free(c);
}

void alpha_fill_s(float *arr, const float num, const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_d(double *arr, const double num, const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_c(ALPHA_Complex8 *arr, const ALPHA_Complex8 num, const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_z(ALPHA_Complex16 *arr, const ALPHA_Complex16 num,
                const size_t size) {
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_fill_random_int(int *arr, unsigned int seed, const size_t size,
                         int upper) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_int(upper);
}
void alpha_fill_random_long(long long *arr, unsigned int seed, const size_t size,
                          long long upper) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_long(upper);
}
void alpha_fill_random_s(float *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_float();
}

void alpha_fill_random_d(double *arr, unsigned int seed, const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  for (size_t i = 0; i < size; ++i) arr[i] = random_double();
}

void alpha_fill_random_c(ALPHA_Complex8 *arr, unsigned int seed,
                       const size_t size) {
  alpha_fill_random_s((float *)arr, seed, size * 2);
}

void alpha_fill_random_z(ALPHA_Complex16 *arr, unsigned int seed,
                       const size_t size) {
  alpha_fill_random_d((double *)arr, seed, size * 2);
}

void alpha_parallel_fill_random_s(float *arr, unsigned int seed,
                                const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < size; ++i) arr[i] = random_float();
}

void alpha_parallel_fill_random_d(double *arr, unsigned int seed,
                                const size_t size) {
  if (seed == 0) seed = time_seed();
  srand(seed);
  ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < size; ++i) arr[i] = random_double();
}

void alpha_parallel_fill_random_c(ALPHA_Complex8 *arr, unsigned int seed,
                                const size_t size) {
  alpha_fill_random_s((float *)arr, seed, size * 2);
}

void alpha_parallel_fill_random_z(ALPHA_Complex16 *arr, unsigned int seed,
                                const size_t size) {
  alpha_fill_random_d((double *)arr, seed, size * 2);
}

void alpha_parallel_fill_s(float *arr, const float num, const size_t size) {
  ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_parallel_fill_d(double *arr, const double num, const size_t size) {
  ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_parallel_fill_c(ALPHA_Complex8 *arr, const ALPHA_Complex8 num,
                         const size_t size) {
  ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}

void alpha_parallel_fill_z(ALPHA_Complex16 *arr, const ALPHA_Complex16 num,
                         const size_t size) {
  ALPHA_INT thread_num = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (size_t i = 0; i < size; ++i) arr[i] = num;
}