#include "alphasparse/spapi.h"
// #include "alphasparse/kernel.h"
#include "doti.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
TYPE ONAME(const ALPHA_INT nz,                              \
                 const TYPE *x,                                 \
                 const ALPHA_INT *indx,                         \
                 const TYPE *y)                                 \
{                                                               \
    return doti(nz, x, indx, y);                          \
}                                                               \

C_IMPL(alphasparse_s_doti, float);   
C_IMPL(alphasparse_d_doti, double);                                        
#undef C_IMPL