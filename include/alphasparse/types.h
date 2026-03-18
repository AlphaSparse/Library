#pragma once

#ifdef __CUDA__
#include "type/r_f16_types.h"
#include "type/r_bf16_types.h"
#include "type/c_f16_types.h"
#include "type/c_bf16_types.h"
#include "type/c_f32_types.h"
#include "type/c_f64_types.h"
#include "type/r_f32_types.h"
#include "type/r_f64_types.h"
#include "type/r_i8_types.h"
#endif
#ifdef __HIP__
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include "type/r_f32_types.h"
#include "type/r_f64_types.h"
#include "type/r_i8_types.h"
#endif
#include <stdint.h>
#define CACHELINE 64
#define F32BYTES 4
#define F64BYTES 8
#define C8BYTES 8
#define C16BYTES 16

#define NPERCLF32 ((CACHELINE) / (F32BYTES))
#define NPERCLF64 ((CACHELINE) / (F64BYTES))
#define NPERCLC8 ((CACHELINE) / (C8BYTES))
#define NPERCLC16 ((CACHELINE) / (C16BYTES))

//如果不同数据类型使用不同的NPERCL,不保证没有bug...目前看来每次处理16列会比较快
//目前只有trsm csr使用了这个宏
#define NPERCL 16

// #ifndef DOUBLE
// #define ALPHA_Float float
// #define ALPHA_Complex ALPHA_Complex8
// #else 
// #define ALPHA_Float double
// #define ALPHA_Complex ALPHA_Complex16
// #endif

// #ifndef COMPLEX
// #define ALPHA_Number ALPHA_Float
// #else 
// #define ALPHA_Number ALPHA_Complex
// #endif

// #ifndef COMPLEX
// #ifndef DOUBLE
// #define ALPHA_Point point_s_t
// #define S S
// // #define NPERCL NPERCLF32
// #else 
// #define ALPHA_Point point_d_t
// #define D D
// // #define NPERCL NPERCLF64
// #endif
// #else
// #ifndef DOUBLE
// #define ALPHA_Point point_c_t
// #define C C
// // #define NPERCL NPERCLC8
// #else 
// #define ALPHA_Point point_z_t
// #define Z Z
// // #define NPERCL NPERCLC16
// #endif
// #endif
 

#ifndef ALPHA_Complex8
typedef
struct {
    float real;
    float imag;
} ALPHA_Complex8;
#endif

#ifndef ALPHA_Complex16
typedef
struct {
    double real;
    double imag;
} ALPHA_Complex16;
#endif

#ifndef ALPHA_INT
    #define ALPHA_INT int32_t
#endif

#ifndef ALPHA_UINT
    #define ALPHA_UINT uint32_t
#endif

#ifndef ALPHA_LONG
    #define ALPHA_LONG int64_t
#endif

#ifndef ALPHA_UINT8
    #define ALPHA_UINT8 uint8_t
#endif

#ifndef ALPHA_INT8
    #define ALPHA_INT8 int8_t
#endif

#ifndef ALPHA_INT16
    #define ALPHA_INT16 int16_t
#endif

#ifndef ALPHA_INT32
    #define ALPHA_INT32 int32_t
#endif

#ifndef ALPHA_INT64
    #define ALPHA_INT64 int64_t
#endif

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
} int2_t;

typedef struct{
    ALPHA_INT x;
    ALPHA_INT y;
    ALPHA_INT z;
} int3_t;

template <typename TYPE>
struct point_t{
    ALPHA_INT x;
    ALPHA_INT y;
    TYPE v;
};

// typedef struct{
//     ALPHA_INT x;
//     ALPHA_INT y;
//     double v;
// } point_d_t;

// typedef struct{
//     ALPHA_INT x;
//     ALPHA_INT y;
//     ALPHA_Complex8 v;
// } point_c_t;

// typedef struct{
//     ALPHA_INT x;
//     ALPHA_INT y;
//     ALPHA_Complex16 v;
// } point_z_t;