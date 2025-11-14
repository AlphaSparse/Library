#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "gthrz.hpp"

#define C_IMPL(ONAME, TYPE)                                     \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                  \
                          TYPE *y,                      \
                          TYPE *x,                      \
                          const ALPHA_INT *indx)                \
{                                                   \
    return gthrz(nz, y, x, indx);                    \
}                                                   \

C_IMPL(alphasparse_s_gthrz, float);   
C_IMPL(alphasparse_d_gthrz, double);   
C_IMPL(alphasparse_c_gthrz, ALPHA_Complex8);   
C_IMPL(alphasparse_z_gthrz, ALPHA_Complex16);                                             
#undef C_IMPL