#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "axpy.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                   \
                          const TYPE a,                         \
                          const TYPE *x,                        \
                          const ALPHA_INT *indx,                \
                          TYPE *y)                               \
{                                                                \
    return axpy(nz, a, x, indx, y);                            \
}                                                                \

C_IMPL(alphasparse_s_axpy, float);   
C_IMPL(alphasparse_d_axpy, double);   
C_IMPL(alphasparse_c_axpy, ALPHA_Complex8);   
C_IMPL(alphasparse_z_axpy, ALPHA_Complex16);                                             
#undef C_IMPL