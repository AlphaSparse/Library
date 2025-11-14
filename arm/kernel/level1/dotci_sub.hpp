#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <stdio.h>
#include "alphasparse/compute.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
void dotci_sub(const IndexType nz,
		   const ValueType *x,
		   const IndexType *indx,
		   const ValueType *y,
		   ValueType *dotci)
{
	ValueType res;
	res = alpha_setzero(res);
	if (nz <= 0)
	{
		fprintf(stderr, "Invalid Values : nz <= 0 !\n");
		return;
	}
	for (IndexType i = 0; i < nz; i++)
	{
		ValueType t;
		t = cmp_conj( x[i]);
		res = alpha_madd(t, y[indx[i]], res);
	}
	*dotci = res;
	return;
}
