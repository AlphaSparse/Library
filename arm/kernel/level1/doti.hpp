#include "alphasparse/kernel.h"
#include <stdio.h>
#include "alphasparse/compute.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
ValueType doti(const IndexType nz,
				const ValueType *x,
				const IndexType *indx,
				const ValueType *y)
{
	ValueType res = 0.f;
	if (nz <= 0)
	{
		fprintf(stderr, "Invalid Values : nz <= 0 !\n");
		return res;
	}

	for (IndexType i = 0; i < nz; i++)
		res += x[i] * y[indx[i]];
	return res;
}
