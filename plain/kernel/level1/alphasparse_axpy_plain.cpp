#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "axpy_plain.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                   \
                          const TYPE a,                         \
                          const TYPE *x,                        \
                          const ALPHA_INT *indx,                \
                          TYPE *y)                               \
{                                                                \
    return axpy_plain(nz, a, x, indx, y);                            \
}                                                                \

C_IMPL(alphasparse_s_axpy_plain, float);   
C_IMPL(alphasparse_d_axpy_plain, double);   
C_IMPL(alphasparse_c_axpy_plain, ALPHA_Complex8);   
C_IMPL(alphasparse_z_axpy_plain, ALPHA_Complex16);                                             
#undef C_IMPL