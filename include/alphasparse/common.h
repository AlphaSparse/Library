#pragma once

#include "alphasparse.h"
#ifdef __CUDA__
#include <cooperative_groups.h>
#endif
#ifdef __HIP__
#include "hip/hip_runtime.h"
#include <hip/hip_cooperative_groups.h>
#endif

__attribute__((unused)) static unsigned int flp2(unsigned int x)
{
  x |= (x >> 1);
  x |= (x >> 2);
  x |= (x >> 4);
  x |= (x >> 8);
  x |= (x >> 16);
  return x - (x >> 1);
}

#ifdef __CUDA__
// find next power of 2
__device__ __host__ __forceinline__ unsigned int
fnp2(unsigned int x)
{
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;

  return x;
}

template <typename T>
__device__ __host__ __forceinline__ T
make_value(float z)
{
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16> || std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>)
  {
    return (T)z;
  }
  else
  {
    return T{z, 0.f};
  }
}
#endif

#ifdef __HIP__

#define CHECK_CUSPARSE(func)                                         \
  {                                                              \
  }
  
template <typename T>
__device__ __host__ __forceinline__ T
make_value(float z)
{
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t> || std::is_same_v<T, int8_t>)
  {
    return (T)z;
  }
  else
  {
    return T{z, 0.f};
  }
}
#endif

#if defined(__CUDA__) || defined(__DCU__) || defined(__HIP__)
__device__ __forceinline__ int32_t alpha_mul24(int32_t x, int32_t y)
{
  return ((x << 8) >> 8) * ((y << 8) >> 8);
}
__device__ __forceinline__ int64_t alpha_mul24(int64_t x, int64_t y)
{
  return ((x << 40) >> 40) * ((y << 40) >> 40);
}
__device__ __forceinline__ int32_t alpha_mad24(int32_t x, int32_t y, int32_t z)
{
  return alpha_mul24(x, y) + z;
}
__device__ __forceinline__ int64_t alpha_mad24(int64_t x, int64_t y, int64_t z)
{
  return alpha_mul24(x, y) + z;
}

template <typename T, typename U>
static inline __device__ U sum2_reduce(U cur_sum, U *partial, int lid, T max_size, int reduc_size)
{
  if (max_size > reduc_size)
  {
    cur_sum = cur_sum + partial[lid + reduc_size];
    // alpha_add(cur_sum, cur_sum, partial[lid + reduc_size]);
    __syncthreads();
    partial[lid] = cur_sum;
  }
  return cur_sum;
}

// Block reduce kernel computing block sum
template <unsigned int BLOCKSIZE, typename T>
__device__ __forceinline__ void alpha_blockreduce_sum(int i, T *data)
{
  if (BLOCKSIZE > 512)
  {
    if (i < 512 && i + 512 < BLOCKSIZE)
    {
      data[i] += data[i + 512];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 256)
  {
    if (i < 256 && i + 256 < BLOCKSIZE)
    {
      data[i] += data[i + 256];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 128)
  {
    if (i < 128 && i + 128 < BLOCKSIZE)
    {
      data[i] += data[i + 128];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 64)
  {
    if (i < 64 && i + 64 < BLOCKSIZE)
    {
      data[i] += data[i + 64];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 32)
  {
    if (i < 32 && i + 32 < BLOCKSIZE)
    {
      data[i] += data[i + 32];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 16)
  {
    if (i < 16 && i + 16 < BLOCKSIZE)
    {
      data[i] += data[i + 16];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 8)
  {
    if (i < 8 && i + 8 < BLOCKSIZE)
    {
      data[i] += data[i + 8];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 4)
  {
    if (i < 4 && i + 4 < BLOCKSIZE)
    {
      data[i] += data[i + 4];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 2)
  {
    if (i < 2 && i + 2 < BLOCKSIZE)
    {
      data[i] += data[i + 2];
    }
    __syncthreads();
  }
  if (BLOCKSIZE > 1)
  {
    if (i < 1 && i + 1 < BLOCKSIZE)
    {
      data[i] += data[i + 1];
    }
    __syncthreads();
  }
}
#endif

template <int WG_SIZE>
static unsigned long long numThreadsForReduction(unsigned long long num_rows)
{
  // #if (defined(__clang__) && __has_builtin(__builtin_clz)) || !defined(__clang) && defined(__GNUG__) && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
  // return (WG_SIZE >> (8 * sizeof(int) - __builtin_clz(num_rows - 1)));
  // #else
  return flp2(WG_SIZE / num_rows);
  // #endif
}

#ifdef __CUDA__
template <unsigned int WFSIZE>
__device__ __forceinline__ float wfreduce_sum(float sum)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, i, WFSIZE);
  return sum;
}

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE, typename T>
__device__ __forceinline__ T wfreduce_sum(T sum)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
    sum += __shfl_xor_sync(0xffffffff, sum, i, WFSIZE);
  return sum;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ cuFloatComplex wfreduce_sum(cuFloatComplex sum)
{
  cuFloatComplex res;
  res.x = wfreduce_sum<WFSIZE>(sum.x);
  res.y = wfreduce_sum<WFSIZE>(sum.y);
  return res;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ cuDoubleComplex wfreduce_sum(cuDoubleComplex sum)
{
  cuDoubleComplex res;
  res.x = wfreduce_sum<WFSIZE>(sum.x);
  res.y = wfreduce_sum<WFSIZE>(sum.y);
  return res;
}

template <typename T>
__device__ __forceinline__ T sub_wfreduce_sum(T sum, int32_t wfsize)
{
  if (wfsize == 2)
    return wfreduce_sum<2>(sum);
  if (wfsize == 4)
    return wfreduce_sum<4>(sum);
  if (wfsize == 8)
    return wfreduce_sum<8>(sum);
  if (wfsize == 16)
    return wfreduce_sum<16>(sum);
  if (wfsize == 32)
    return wfreduce_sum<32>(sum);
  if (wfsize == 64)
    return wfreduce_sum<64>(sum);
  return T{};
}

template <unsigned int WFSIZE>
__device__ __forceinline__ float wfreduce_min(float min_)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
  {
    float t_min = __shfl_xor_sync(0xffffffff, min_, i, WFSIZE);
    min_ = min(min_, t_min);
  }
  return min_;
}

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE, typename T>
__device__ __forceinline__ T wfreduce_min(T min_)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
  {
    T t_min = __shfl_xor_sync(0xffffffff, min_, i, WFSIZE);
    min_ = min(min_, t_min);
  }
  return min_;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ cuFloatComplex wfreduce_min(cuFloatComplex min_)
{
  cuFloatComplex res;
  res.x = wfreduce_min<WFSIZE>(min_.x);
  res.y = wfreduce_min<WFSIZE>(min_.y);
  return res;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ cuDoubleComplex wfreduce_min(cuDoubleComplex min_)
{
  cuDoubleComplex res;
  res.x = wfreduce_min<WFSIZE>(min_.x);
  res.y = wfreduce_min<WFSIZE>(min_.y);
  return res;
}

using cooperative_groups::this_thread_block;
using cooperative_groups::thread_block_tile;

template <unsigned Size, typename Group>
__device__ __forceinline__ thread_block_tile<Size, void> tiled_partition(
    Group &g)
{
  return cooperative_groups::tiled_partition<Size>(g);
}

template <unsigned subwarp_size = 32, typename ValueType, typename IndexType>
__device__ __forceinline__ bool segment_scan(
    const cooperative_groups::thread_block_tile<subwarp_size> &group,
    const IndexType ind,
    ValueType &val)
{
  bool head = true;
#pragma unroll
  for (int i = 1; i < subwarp_size; i <<= 1)
  {
    const IndexType add_ind = group.shfl_up(ind, i);
    ValueType add_val{};
    if (add_ind == ind && group.thread_rank() >= i)
    {
      add_val = val;
      if (i == 1)
      {
        head = false;
      }
    }
    add_val = group.shfl_down(add_val, i);
    if (group.thread_rank() < subwarp_size - i)
    {
      val += add_val;
    }
  }
  return head;
}

template <typename T, typename V, typename W>
__global__ void array_scale(T m, V *array, W beta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    array[idx] *= beta;
  }
}

template <typename T>
__host__ __device__ __forceinline__ T ceildivT(const T nom, const T denom)
{
  return (nom + denom - 1ll) / denom;
}

// types common kernels
#define FULL_MASK 0xffffffff
template <unsigned int WFSIZE>
__device__ __forceinline__ float alpha_reduce_sum(float sum)
{
  for (int offset = WFSIZE/2; offset > 0; offset /= 2)
      sum += __shfl_down_sync(FULL_MASK, sum, offset);

  return sum;
}
#undef FULL_MASK

#define FULL_MASK 0xffffffff
template <unsigned int WFSIZE>
__device__ __forceinline__ double alpha_reduce_sum(double sum)
{
  for (int offset = WFSIZE/2; offset > 0; offset /= 2)
      sum += __shfl_down_sync(FULL_MASK, sum, offset);

  return sum;
}
#undef FULL_MASK

__device__ __forceinline__ cuFloatComplex
atomicAdd(cuFloatComplex* lh, const cuFloatComplex& rh)
{
  atomicAdd(&(lh->x), rh.x);
  atomicAdd(&(lh->y), rh.y);
  return *lh;
}

__device__ __forceinline__ cuFloatComplex
__shfl_down_sync(unsigned int mask, cuFloatComplex& var, unsigned int delta, int width = 32)
{
  cuFloatComplex tmp {};
  tmp.x = __shfl_down_sync(mask, var.x, delta, width);
  tmp.y = __shfl_down_sync(mask, var.y, delta, width);
  return tmp;
}

#define FULL_MASK 0xffffffff
template<unsigned int WFSIZE>
__device__ __forceinline__ cuFloatComplex
alpha_reduce_sum(cuFloatComplex& sum)
{
  cuFloatComplex r;

  float x = sum.x;
  float y = sum.y;

  for (int offset = WFSIZE / 2; offset > 0; offset /= 2) {
    x += __shfl_down_sync(FULL_MASK, x, offset);
    y += __shfl_down_sync(FULL_MASK, y, offset);
  }
  r = make_cuFloatComplex(x, y);
  return r;
}
#undef FULL_MASK

__device__ __forceinline__ cuDoubleComplex
atomicAdd(cuDoubleComplex* lh, const cuDoubleComplex& rh)
{
  atomicAdd(&(lh->x), rh.x);
  atomicAdd(&(lh->y), rh.y);
  return *lh;
}

__device__ __forceinline__ cuDoubleComplex
__shfl_down_sync(unsigned int mask, cuDoubleComplex& var, unsigned int delta, int width = 32)
{
  cuDoubleComplex tmp {};
  tmp.x = __shfl_down_sync(mask, var.x, delta, width);
  tmp.y = __shfl_down_sync(mask, var.y, delta, width);
  return tmp;
}

#define FULL_MASK 0xffffffff
template <unsigned int WFSIZE>
__device__ __forceinline__ cuDoubleComplex alpha_reduce_sum(cuDoubleComplex sum)
{
  cuDoubleComplex r;

  double x = sum.x;
  double y = sum.y;

  for (int offset = WFSIZE/2; offset > 0; offset /= 2)
  {
      x += __shfl_down_sync(FULL_MASK, x, offset);
      y += __shfl_down_sync(FULL_MASK, y, offset);
  }

  r = make_cuDoubleComplex(x, y);
  return r;
}
#undef FULL_MASK

__device__ __forceinline__ half2
atomicAdd(half2* lh, const cuFloatComplex& rh)
{
  atomicAdd(&(lh->x), rh.x);
  atomicAdd(&(lh->y), rh.y);
  return *lh;
}
#endif

#ifdef __HIP__
template <unsigned int WFSIZE>
__device__ __forceinline__ float wfreduce_sum(float sum)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
    sum += __shfl_xor(sum, i, WFSIZE);
  return sum;
}

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE, typename T>
__device__ __forceinline__ T wfreduce_sum(T sum)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
    sum += __shfl_xor(sum, i, WFSIZE);
  return sum;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ hipFloatComplex wfreduce_sum(hipFloatComplex sum)
{
  hipFloatComplex res;
  res.x = wfreduce_sum<WFSIZE>(sum.x);
  res.y = wfreduce_sum<WFSIZE>(sum.y);
  return res;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ hipDoubleComplex wfreduce_sum(hipDoubleComplex sum)
{
  hipDoubleComplex res;
  res.x = wfreduce_sum<WFSIZE>(sum.x);
  res.y = wfreduce_sum<WFSIZE>(sum.y);
  return res;
}

template <typename T>
__device__ __forceinline__ T sub_wfreduce_sum(T sum, int32_t wfsize)
{
  if (wfsize == 2)
    return wfreduce_sum<2>(sum);
  if (wfsize == 4)
    return wfreduce_sum<4>(sum);
  if (wfsize == 8)
    return wfreduce_sum<8>(sum);
  if (wfsize == 16)
    return wfreduce_sum<16>(sum);
  if (wfsize == 32)
    return wfreduce_sum<32>(sum);
  if (wfsize == 64)
    return wfreduce_sum<64>(sum);
  return T{};
}

template <unsigned int WFSIZE>
__device__ __forceinline__ float wfreduce_min(float min_)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
  {
    float t_min = __shfl_xor(min_, i, WFSIZE);
    min_ = min(min_, t_min);
  }
  return min_;
}

// DPP-based float wavefront reduction sum
template <unsigned int WFSIZE, typename T>
__device__ __forceinline__ T wfreduce_min(T min_)
{
  for (int i = WFSIZE / 2; i >= 1; i /= 2)
  {
    T t_min = __shfl_xor(min_, i, WFSIZE);
    min_ = min(min_, t_min);
  }
  return min_;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ hipFloatComplex wfreduce_min(hipFloatComplex min_)
{
  hipFloatComplex res;
  res.x = wfreduce_min<WFSIZE>(min_.x);
  res.y = wfreduce_min<WFSIZE>(min_.y);
  return res;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ hipDoubleComplex wfreduce_min(hipDoubleComplex min_)
{
  hipDoubleComplex res;
  res.x = wfreduce_min<WFSIZE>(min_.x);
  res.y = wfreduce_min<WFSIZE>(min_.y);
  return res;
}

using cooperative_groups::this_thread_block;
using cooperative_groups::thread_block_tile;

template <unsigned Size, typename Group>
__device__ __forceinline__ thread_block_tile<Size, void> tiled_partition(
    Group &g)
{
  return cooperative_groups::tiled_partition<Size>(g);
}

template <unsigned subwarp_size = 32, typename ValueType, typename IndexType>
__device__ __forceinline__ bool segment_scan(
    const cooperative_groups::thread_block_tile<subwarp_size> &group,
    const IndexType ind,
    ValueType &val)
{
  bool head = true;
#pragma unroll
  for (int i = 1; i < subwarp_size; i <<= 1)
  {
    const IndexType add_ind = group.shfl_up(ind, i);
    ValueType add_val{};
    if (add_ind == ind && group.thread_rank() >= i)
    {
      add_val = val;
      if (i == 1)
      {
        head = false;
      }
    }
    add_val = group.shfl_down(add_val, i);
    if (group.thread_rank() < subwarp_size - i)
    {
      val += add_val;
    }
  }
  return head;
}

template <typename T, typename V, typename W>
__global__ void array_scale(T m, V *array, W beta)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < m)
  {
    array[idx] *= beta;
  }
}

template <typename T>
__host__ __device__ __forceinline__ T ceildivT(const T nom, const T denom)
{
  return (nom + denom - 1ll) / denom;
}




#define GPU_TIMER_START(elapsed_time, event_start, event_stop) \
  do                                                           \
  {                                                            \
    elapsed_time = 0.0;                                        \
    hipEventCreate(&event_start);                             \
    hipEventCreate(&event_stop);                              \
    hipEventRecord(event_start);                              \
  } while (0)

#define GPU_TIMER_END(elapsed_time, event_start, event_stop)      \
  do                                                              \
  {                                                               \
    hipEventRecord(event_stop);                                  \
    hipEventSynchronize(event_stop);                             \
    hipEventElapsedTime(&elapsed_time, event_start, event_stop); \
  } while (0)

#define CHECK_CUDA(func)                                         \
  {                                                              \
    hipError_t status = (func);                                 \
    if (status != hipSuccess)                                   \
    {                                                            \
      printf("CUDA API failed at line %d with error: %s (%d)\n", \
             __LINE__,                                           \
             hipGetErrorString(status),                         \
             status);                                            \
      exit(-1);                                                  \
    }                                                            \
  }

__device__ __forceinline__ float
alphasparse_fma(float p,
                float q,
                float r)
{
    return fma(p, q, r);
}

__device__ __forceinline__ double
alphasparse_fma(double p,
                double q,
                double r)
{
    return fma(p, q, r);
}

__device__ __forceinline__ hipFloatComplex
alphasparse_fma(hipFloatComplex p,
                hipFloatComplex q,
                hipFloatComplex r)
{
    return {};
}

__device__ __forceinline__ hipDoubleComplex
alphasparse_fma(hipDoubleComplex p,
                hipDoubleComplex q,
                hipDoubleComplex r)
{
    return {};
}




template <unsigned int WFSIZE>
__device__ __forceinline__ void alphasparse_wfreduce_sum(int *sum)
{
    for (int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *sum += __shfl_xor(*sum, i);
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ void alphasparse_wfreduce_sum(int64_t *sum)
{
    for (int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        *sum += __shfl_xor(*sum, i);
    }
}

template <unsigned int WFSIZE>
__device__ __forceinline__ float alphasparse_wfreduce_sum(float sum)
{
    for (int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

template <unsigned int WFSIZE>
__device__ __forceinline__ double alphasparse_wfreduce_sum(double sum)
{
    for (int i = WFSIZE >> 1; i > 0; i >>= 1)
    {
        sum += __shfl_xor(sum, i);
    }

    return sum;
}

#endif
