#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "gthrz_plain.hpp"

#define C_IMPL(ONAME, TYPE)                                     \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                  \
                          TYPE *y,                      \
                          TYPE *x,                      \
                          const ALPHA_INT *indx)                \
{                                                   \
    return gthrz_plain(nz, y, x, indx);                    \
}                                                   \

C_IMPL(alphasparse_s_gthrz_plain, float);   
C_IMPL(alphasparse_d_gthrz_plain, double);   
C_IMPL(alphasparse_c_gthrz_plain, ALPHA_Complex8);   
C_IMPL(alphasparse_z_gthrz_plain, ALPHA_Complex16);                                             
#undef C_IMPL