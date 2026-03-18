#include "alphasparse/spapi_plain.h"
#include "alphasparse/kernel_plain.h"
#include "dotci_sub_plain.hpp"

#define C_IMPL(ONAME, TYPE)   						\
void ONAME(const ALPHA_INT nz,						\
		   const TYPE *x,							\
		   const ALPHA_INT *indx,					\
		   const TYPE *y,							\
		   TYPE *dutci)								\
{															\
	return dotci_sub_plain(nz, x, indx, y, dutci);				\
}															\

C_IMPL(alphasparse_c_dotci_sub_plain, ALPHA_Complex8);   
C_IMPL(alphasparse_z_dotci_sub_plain, ALPHA_Complex16);                                               
#undef C_IMPL