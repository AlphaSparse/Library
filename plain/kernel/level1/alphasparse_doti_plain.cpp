#include "alphasparse/spapi_plain.h"
// #include "alphasparse/kernel_plain.h"
#include "doti_plain.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
TYPE ONAME(const ALPHA_INT nz,                              \
                 const TYPE *x,                                 \
                 const ALPHA_INT *indx,                         \
                 const TYPE *y)                                 \
{                                                               \
    return doti_plain(nz, x, indx, y);                          \
}                                                               \

C_IMPL(alphasparse_s_doti_plain, float);   
C_IMPL(alphasparse_d_doti_plain, double);                                        
#undef C_IMPL