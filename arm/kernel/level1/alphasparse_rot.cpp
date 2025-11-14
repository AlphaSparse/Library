#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "rot.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                    \
                          TYPE *x,                               \
                          const ALPHA_INT *indx,                 \
                          TYPE *y,                               \
                          const TYPE c,                          \
                          const TYPE s)                          \
{                                                       \
    return rot(nz, x, indx, y, c, s);                 \
}                                                       \

C_IMPL(alphasparse_s_rot, float);   
C_IMPL(alphasparse_d_rot, double);                                        
#undef C_IMPL