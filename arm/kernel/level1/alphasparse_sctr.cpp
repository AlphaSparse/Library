#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "sctr.hpp"

#define C_IMPL(ONAME, TYPE)                                      \
alphasparseStatus_t ONAME(const ALPHA_INT nz,                   \
                          const TYPE *x,                        \
                          const ALPHA_INT *indx,                \
                          TYPE *y)                              \
{                                                       \
    return sctr(nz, x, indx, y);                                        \
}                                                       \

C_IMPL(alphasparse_s_sctr, float);   
C_IMPL(alphasparse_d_sctr, double);   
C_IMPL(alphasparse_c_sctr, ALPHA_Complex8);   
C_IMPL(alphasparse_z_sctr, ALPHA_Complex16);                                             
#undef C_IMPL