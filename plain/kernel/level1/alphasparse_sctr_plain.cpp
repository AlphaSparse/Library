#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "sctr_plain.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                   \
                          const TYPE *x,                        \
                          const ALPHA_INT *indx,                \
                          TYPE *y)                              \
{                                                       \
    return sctr_plain(nz, x, indx, y);                                        \
}                                                       \

C_IMPL(alphasparse_s_sctr_plain, float);   
C_IMPL(alphasparse_d_sctr_plain, double);   
C_IMPL(alphasparse_c_sctr_plain, ALPHA_Complex8);   
C_IMPL(alphasparse_z_sctr_plain, ALPHA_Complex16);                                             
#undef C_IMPL