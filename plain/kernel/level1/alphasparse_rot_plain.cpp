#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "rot_plain.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                    \
                          TYPE *x,                               \
                          const ALPHA_INT *indx,                 \
                          TYPE *y,                               \
                          const TYPE c,                          \
                          const TYPE s)                          \
{                                                       \
    return rot_plain(nz, x, indx, y, c, s);                 \
}                                                       \

C_IMPL(alphasparse_s_rot_plain, float);   
C_IMPL(alphasparse_d_rot_plain, double);                                        
#undef C_IMPL