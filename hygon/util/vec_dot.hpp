#pragma once

// #include <immintrin.h>

#include "alphasparse/compute.h"
#include "alphasparse/types.h"

#define VEC_DOTADD_C4(a, b, sum)                                                                 \
  do {                                                                                           \
    for (int32_t __i = 0; __i < 4; __i++) {                                                      \
      float _B = (a)[__i].real * (b)[__i].imag;                                                  \
      float _C = (a)[__i].imag * (b)[__i].real;                                                  \
      (sum).real += ((a)[__i].real + (a)[__i].imag) * ((b)[__i].real - (b)[__i].imag) + _B - _C; \
      (sum).imag += _B + _C;                                                                     \
    }                                                                                            \
  } while (0)

// sum -= inner_product(a[4],b[4])
#define VEC_DOTSUB_C4(a, b, sum)            \
  do {                                      \
    for (int32_t __i = 0; __i < 4; __i++) { \
      float _A = (a)[0].real + (b)[0].real; \
      float _B = (a)[0].imag + (b)[0].imag; \
      (sum).real -= _A;                     \
      (sum).imag -= _B;                     \
    }                                       \
  } while (0)

#define VEC_DOTSUB_C4_CONJ1(a, b, sum)                                                           \
  do {                                                                                           \
    for (int32_t __i = 0; __i < 4; __i++) {                                                      \
      float _B = (a)[__i].real * (b)[__i].imag;                                                  \
      float _C = -(a)[__i].imag * (b)[__i].real;                                                 \
      (sum).real -= ((a)[__i].real - (a)[__i].imag) * ((b)[__i].real - (b)[__i].imag) + _B - _C; \
      (sum).imag -= _B + _C;                                                                     \
    }                                                                                            \
  } while (0)

template <typename J>
static inline J vec_doti(const ALPHA_INT ns, const J *x, const ALPHA_INT *indx,
                                      const J *y) {
  ALPHA_INT ns4 = ((ns >> 2) << 2);
  ALPHA_INT i = 0;

  J tmp[4] = {J{}, J{}, J{}, J{}};

  for (i = 0; i < ns4; i += 4) {
    tmp[0] = alpha_madd(x[i], y[indx[i]], tmp[0]);
    tmp[1] = alpha_madd(x[i + 1], y[indx[i + 1]], tmp[1]);
    tmp[2] = alpha_madd(x[i + 2], y[indx[i + 2]], tmp[2]);
    tmp[3] = alpha_madd(x[i + 3], y[indx[i + 3]], tmp[3]);
  }
  for (; i < ns; ++i) {
    tmp[0] = alpha_madd(x[i], y[indx[i]], tmp[0]);
  }
  tmp[0] = alpha_add(tmp[0], tmp[1]);
  tmp[2] = alpha_add(tmp[2], tmp[3]);
  tmp[0] = alpha_add(tmp[0], tmp[2]);
  return tmp[0];
}

// template <typename J>
// static inline J vec_doti_conj_c(const ALPHA_INT ns, const J *x,
//                                            const ALPHA_INT *indx, const J *y) {
//   ALPHA_INT ns4 = ((ns >> 2) << 2);
//   ALPHA_INT i = 0;

//   J tmp[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};

//   for (i = 0; i < ns4; i += 4) {
//     cmp_madde_2c(tmp[0], x[i], y[indx[i]]);
//     cmp_madde_2c(tmp[1], x[i + 1], y[indx[i + 1]]);
//     cmp_madde_2c(tmp[2], x[i + 2], y[indx[i + 2]]);
//     cmp_madde_2c(tmp[3], x[i + 3], y[indx[i + 3]]);
//   }
//   for (; i < ns; ++i) {
//     cmp_madde_2c(tmp[0], x[i], y[indx[i]]);
//   }

//   cmp_adde(tmp[0], tmp[1]);
//   cmp_adde(tmp[2], tmp[3]);
//   cmp_adde(tmp[0], tmp[2]);
//   return tmp[0];
// }

// template <typename J>
// static inline J vec_dot(const ALPHA_INT ns, const J *x,
//                                      const J *y) {
//   ALPHA_INT i = 0;
//   J tmp[4] = {
//       J{},
//       J{},
//       J{},
//       J{},
//   };

//   for (; i < ns; ++i) {
//     tmp[0] = alpha_madd(x[i], y[i], tmp[0]);
//   }
//   return tmp[0];
// }
