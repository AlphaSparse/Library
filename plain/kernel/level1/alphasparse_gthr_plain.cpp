#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "gthr_plain.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                   \
                          const TYPE *y,                 \
                          TYPE   *x,                       \
                          const ALPHA_INT *indx)                  \
{                                                   \
    return gthr_plain(nz, y, x, indx);                  \
}                                                   \

C_IMPL(alphasparse_s_gthr_plain, float);   
C_IMPL(alphasparse_d_gthr_plain, double);   
C_IMPL(alphasparse_c_gthr_plain, ALPHA_Complex8);   
C_IMPL(alphasparse_z_gthr_plain, ALPHA_Complex16);                                             
#undef C_IMPL