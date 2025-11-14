#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "dotci_sub.hpp"

#define C_IMPL(ONAME, TYPE)   						\
void ONAME(const ALPHA_INT nz,						\
		   const TYPE *x,							\
		   const ALPHA_INT *indx,					\
		   const TYPE *y,							\
		   TYPE *dutci)								\
{															\
	return dotci_sub(nz, x, indx, y, dutci);				\
}															\

C_IMPL(alphasparse_c_dotci_sub, ALPHA_Complex8);   
C_IMPL(alphasparse_z_dotci_sub, ALPHA_Complex16);                                               
#undef C_IMPL