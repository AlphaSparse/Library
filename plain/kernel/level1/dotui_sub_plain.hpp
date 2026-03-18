#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include <stdio.h>


template <typename IndexType = ALPHA_INT, typename ValueType>
void dotui_sub_plain(const IndexType nz,
		   const ValueType *x,
		   const IndexType *indx,
		   const ValueType *y,
		   ValueType *dotui)
{
	ValueType res;
	alpha_setzero(res);
	if (nz <= 0)
	{
		fprintf(stderr, "Invalid Values : nz <= 0 !\n");
		return;
	}
	for (IndexType i = 0; i < nz; i++)
		res = alpha_madd(x[i], y[indx[i]], res);
	(*dotui) = res;
	return;
}
