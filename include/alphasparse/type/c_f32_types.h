/* ************************************************************************
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#ifndef C_F32_TYPES_H
#define C_F32_TYPES_H

#include <cuComplex.h>
#include <iostream>
#include <assert.h>

/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ cuFloatComplex
operator+(const cuFloatComplex& lh, const cuFloatComplex& rh)
{
  return cuCaddf(lh, rh);
}
__device__ __forceinline__ cuFloatComplex
operator-(const cuFloatComplex& lh, const cuFloatComplex& rh)
{
  return cuCsubf(lh, rh);
}
__device__ __forceinline__ cuFloatComplex
operator*(const cuFloatComplex& lh, const cuFloatComplex& rh)
{
  return cuCmulf(lh, rh);
}
__device__ __forceinline__ cuFloatComplex
operator/(const cuFloatComplex& lh, const cuFloatComplex& rh)
{
  return cuCdivf(lh, rh);
}

__device__ __forceinline__ cuFloatComplex&
operator+=(cuFloatComplex& lh, const cuFloatComplex& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator-=(cuFloatComplex& lh, const cuFloatComplex& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator*=(cuFloatComplex& lh, const cuFloatComplex& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ cuFloatComplex&
operator/=(cuFloatComplex& lh, const cuFloatComplex& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ cuFloatComplex
operator*(const cuFloatComplex& lh, const int32_t& rh)
{
  return make_cuFloatComplex(lh.x * rh, lh.y * rh);
}
__device__ __forceinline__ cuFloatComplex
operator*(const int32_t& lh, const cuFloatComplex& rh)
{
  return make_cuFloatComplex(rh.x * lh, rh.y * lh);
}
__device__ __forceinline__ cuFloatComplex
operator*(const cuFloatComplex& lh, const int64_t& rh)
{
  return make_cuFloatComplex(lh.x * rh, lh.y * rh);
}
__device__ __forceinline__ cuFloatComplex
operator*(const int64_t& lh, const cuFloatComplex& rh)
{
  return make_cuFloatComplex(rh.x * lh, rh.y * lh);
}

__host__ __device__ __forceinline__ bool
operator==(const cuFloatComplex& lh, const cuFloatComplex& rh)
{
  return lh.x == rh.x && lh.y == rh.y;
}

__host__ __device__ __forceinline__ bool
operator!=(const cuFloatComplex& lh, const cuFloatComplex& rh)
{
  return lh.x != rh.x || lh.y != rh.y;
}

__device__ __forceinline__ cuFloatComplex
operator-(const cuFloatComplex& z)
{
  return make_cuFloatComplex(-z.x, -z.y);
}

__device__ __forceinline__ cuFloatComplex
conj(const cuFloatComplex& z)
{
  return cuConjf(z);
}

__host__ __device__ __forceinline__ bool
is_zero(const cuFloatComplex& z)
{
  return z.x == 0.f && z.y == 0.f;
}

__device__ __forceinline__ float
alpha_abs(const cuFloatComplex& z)
{
  float real = alpha_abs(z.x);
  float imag = alpha_abs(z.y);

  if(real > imag)
    return (imag /= real, real * alpha_sqrt(imag * imag + 1));
  else if(imag)
      return (real /= imag, imag * alpha_sqrt(real * real + 1));
    else 
      return 0;

  // return real > imag ? (imag /= real, real * alpha_sqrt(imag * imag + 1))
  //        : imag      ? (real /= imag, imag * alpha_sqrt(real * real + 1))
  //                    : 0;
  // // cuFloatComplex r = make_cuFloatComplex (sqrt(z.x * z.x + z.y * z.y),
  // 0.0f);
  // // cuFloatComplex r = make_cuFloatComplex (sqrtf(z.x * z.x), 0.0f);
  // cuFloatComplex r = make_cuFloatComplex (fabsf(z.x), 0.0f);
  // return r;
}

__device__ __forceinline__ cuFloatComplex
alpha_sqrt(const cuFloatComplex& z)
{
  float x = z.x;
  float y = z.y;

  float sgnp = (y < 0.0f) ? -1.0f : 1.0f;
  float absz = alpha_abs(z);

  return make_cuFloatComplex(sqrt((absz + x) * 0.5f),
                             sgnp * sqrt((absz - x) * 0.5f));
  // cuFloatComplex r = make_cuFloatComplex(sqrtf(z.x), 0.0f);
  // return r;
}

__device__ __forceinline__ bool alpha_gt(const cuFloatComplex& x, const cuFloatComplex& y)
{
    if(&x == &y)
    {
        return false;
    }

    assert(x.y == y.y && x.y == 0.0f);

    return x.x > y.x;
}

template <typename T>
__device__ __forceinline__ T
alpha_cast(const cuFloatComplex& z)
{
  T r = make_cuFloatComplex (z.x, 0.0f);
  return r;
}

__device__ __forceinline__ cuFloatComplex
alpha_fma(cuFloatComplex p, cuFloatComplex q, cuFloatComplex r)
{
  float real = alpha_fma(-p.y, q.y, alpha_fma(p.x, q.x, r.x));
  float imag = alpha_fma(p.x, q.y, alpha_fma(p.y, q.x, r.y));
  return make_cuFloatComplex(real, imag);
}

#endif /* C_F32_TYPES_H */
