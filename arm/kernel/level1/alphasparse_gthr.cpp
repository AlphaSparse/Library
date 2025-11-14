#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "gthr.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                   \
                          const TYPE *y,                 \
                          TYPE   *x,                       \
                          const ALPHA_INT *indx)                  \
{                                                   \
    return gthr(nz, y, x, indx);                  \
}                                                   \

C_IMPL(alphasparse_s_gthr, float);   
C_IMPL(alphasparse_d_gthr, double);   
C_IMPL(alphasparse_c_gthr, ALPHA_Complex8);   
C_IMPL(alphasparse_z_gthr, ALPHA_Complex16);                                             
#undef C_IMPL