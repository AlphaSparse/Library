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



#ifndef C_F64_TYPES_H
#define C_F64_TYPES_H

#include <cuComplex.h>
#include <assert.h>

/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ cuDoubleComplex
operator+(const cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  return cuCadd(lh, rh);
}
__device__ __forceinline__ cuDoubleComplex
operator-(const cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  return cuCsub(lh, rh);
}
__device__ __forceinline__ cuDoubleComplex
operator*(const cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  return cuCmul(lh, rh);
}
__device__ __forceinline__ cuDoubleComplex
operator/(const cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  return cuCdiv(lh, rh);
}

__device__ __forceinline__ cuDoubleComplex&
operator+=(cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  lh = lh + rh;
  return lh;
}
__device__ __forceinline__ cuDoubleComplex&
operator-=(cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  lh = lh - rh;
  return lh;
}
__device__ __forceinline__ cuDoubleComplex&
operator*=(cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  lh = lh * rh;
  return lh;
}
__device__ __forceinline__ cuDoubleComplex&
operator/=(cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  lh = lh / rh;
  return lh;
}

__device__ __forceinline__ cuDoubleComplex
operator*(const cuDoubleComplex& lh, const int32_t& rh)
{
  return make_cuDoubleComplex(lh.x * rh, lh.y * rh);
}
__device__ __forceinline__ cuDoubleComplex
operator*(const int32_t& lh, const cuDoubleComplex& rh)
{
  return make_cuDoubleComplex(rh.x * lh, rh.y * lh);
}
__device__ __forceinline__ cuDoubleComplex
operator*(const cuDoubleComplex& lh, const int64_t& rh)
{
  return make_cuDoubleComplex(lh.x * rh, lh.y * rh);
}
__device__ __forceinline__ cuDoubleComplex
operator*(const int64_t& lh, const cuDoubleComplex& rh)
{
  return make_cuDoubleComplex(rh.x * lh, rh.y * lh);
}

__host__ __device__ __forceinline__ bool
operator==(const cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  return lh.x == rh.x && lh.y == rh.y;
}

__host__ __device__ __forceinline__ bool
operator!=(const cuDoubleComplex& lh, const cuDoubleComplex& rh)
{
  return lh.x != rh.x || lh.y != rh.y;
}

__device__ __forceinline__ cuDoubleComplex
operator-(const cuDoubleComplex& z)
{
  return make_cuDoubleComplex(-z.x, -z.y);
}

__device__ __forceinline__ cuDoubleComplex
conj(const cuDoubleComplex& z)
{
  return cuConj(z);
}

__host__ __device__ __forceinline__ bool
is_zero(const cuDoubleComplex& z)
{
  return z.x == 0.f && z.y == 0.f;
}

__device__ __forceinline__ double
alpha_abs(const cuDoubleComplex& z)
{
  double real = alpha_abs(z.x);
  double imag = alpha_abs(z.y);

  // return real > imag ? (imag /= real, real * alpha_sqrt(imag * imag + 1))
  //         : imag      ? (real /= imag, imag * alpha_sqrt(real * real + 1))
  //                     : 0;
  if(real > imag)
    return (imag /= real, real * alpha_sqrt(imag * imag + 1));
  else if(imag)
      return (real /= imag, imag * alpha_sqrt(real * real + 1));
    else 
      return 0;
  // // cuDoubleComplex r = make_cuDoubleComplex(sqrt(z.x * z.x + z.y * z.y), 0.0f);//rocSPARSE style
  // // cuDoubleComplex r = make_cuDoubleComplex(sqrt(z.x * z.x), 0.0f);//cuSPARSE style
  // cuDoubleComplex r = make_cuDoubleComplex(fabs(z.x), 0.0);//cuSPARSE style
  // return r;
}

__device__ __forceinline__ cuDoubleComplex
alpha_sqrt(const cuDoubleComplex& z)
{
  double x = z.x;
  double y = z.y;

  double sgnp = (y < 0.0f) ? -1.0f : 1.0f;
  double absz = alpha_abs(z);

  return make_cuDoubleComplex(sqrt((absz + x) * 0.5f), sgnp * sqrt((absz - x) * 0.5f));
  // cuDoubleComplex r = make_cuDoubleComplex (sqrt(z.x), 0.0);
  // return r;
}

__device__ __forceinline__ bool alpha_gt(const cuDoubleComplex& x, const cuDoubleComplex& y)
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
alpha_cast(const cuDoubleComplex& z)
{
  T r = make_cuDoubleComplex (z.x, 0.0f);
  return r;
}

__device__ __forceinline__ cuDoubleComplex alpha_fma(cuDoubleComplex p, cuDoubleComplex q, cuDoubleComplex r){
  double real = alpha_fma(-p.y, q.y, alpha_fma(p.x, q.x, r.x));
  double imag = alpha_fma(p.x, q.y, alpha_fma(p.y, q.x, r.y));
  return make_cuDoubleComplex(real, imag);
}

#endif /* C_F64_TYPES_H */
