#pragma once

#include <hip/hip_runtime.h>
#include "alphasparse/common.h"
#include "alphasparse/types.h"
#include <hip/hip_runtime.h>


__host__ __device__ __forceinline__ hipFloatComplex
atomicAdd(hipFloatComplex *lh, const hipFloatComplex &rh)
{
  atomicAdd(&(lh->x), rh.x);
  atomicAdd(&(lh->y), rh.y);
  return *lh;
}

template <unsigned int WFSIZE>
__host__ __device__ __forceinline__ hipFloatComplex
alpha_reduce_sum(hipFloatComplex &sum)
{
  hipFloatComplex r;

  float x = sum.x;
  float y = sum.y;

  for (int offset = WFSIZE / 2; offset > 0; offset /= 2)
  {
    x += __shfl_down(x, offset);
    y += __shfl_down(y, offset);
  }
  r = make_hipFloatComplex(x, y);
  return r;
}

__device__ inline hipFloatComplex __shfl_down(hipFloatComplex var, unsigned int lane_delta, int width = 64)
{
  hipFloatComplex r;
  float x = __shfl_down(var.x, lane_delta, width);
  float y = __shfl_down(var.y, lane_delta, width);
  r = make_hipFloatComplex(x, y);
  return r;
}

__host__ __device__ __forceinline__ hipDoubleComplex
atomicAdd(hipDoubleComplex *lh, const hipDoubleComplex &rh)
{
  atomicAdd(&(lh->x), rh.x);
  atomicAdd(&(lh->y), rh.y);
  return *lh;
}

template <unsigned int WFSIZE>
__host__ __device__ __forceinline__ hipDoubleComplex alpha_reduce_sum(hipDoubleComplex sum)
{
  hipDoubleComplex r;

  double x = sum.x;
  double y = sum.y;

  for (int offset = WFSIZE / 2; offset > 0; offset /= 2)
  {
    x += __shfl_down(x, offset);
    y += __shfl_down(y, offset);
  }

  r = make_hipDoubleComplex(x, y);
  return r;
}

__device__ inline hipDoubleComplex __shfl_down(hipDoubleComplex var, unsigned int lane_delta, int width = 64)
{
  hipDoubleComplex r;
  double x = __shfl_down(var.x, lane_delta, width);
  double y = __shfl_down(var.y, lane_delta, width);
  r = make_hipDoubleComplex(x, y);
  return r;
}

template <unsigned int WFSIZE>
__host__ __device__ __forceinline__ float alpha_reduce_sum(float sum)
{
  for (int offset = WFSIZE/2; offset > 0; offset /= 2)
      sum += __shfl_down(sum, offset);

  return sum;
}

template <unsigned int WFSIZE>
__host__ __device__ __forceinline__ double alpha_reduce_sum(double sum)
{
  for (int offset = WFSIZE/2; offset > 0; offset /= 2)
      sum += __shfl_down(sum, offset);

  return sum;
}
