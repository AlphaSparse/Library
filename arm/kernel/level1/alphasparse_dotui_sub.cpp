#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "dotui_sub.hpp"

#define C_IMPL(ONAME, TYPE)				\
void ONAME(const ALPHA_INT nz,			\
		   const TYPE *x,				\
		   const ALPHA_INT *indx,		\
		   const TYPE *y,				\
		   TYPE *dutci)					\
{													\
	dotui_sub(nz, x, indx, y, dutci);		\
}													\

C_IMPL(dotui_c_sub, ALPHA_Complex8);   
C_IMPL(dotui_z_sub, ALPHA_Complex16);                                             
#undef C_IMPL