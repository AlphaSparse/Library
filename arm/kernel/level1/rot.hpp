#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
alphasparseStatus_t rot(const IndexType nz,
	  ValueType *x,
	  const IndexType *indx,
	  ValueType *y,
	  const ValueType c,
	  const ValueType s)
{
	for (IndexType i = 0; i < nz; ++i)
	{
		ValueType t = x[i];
		x[i] = c * x[i] + s * y[indx[i]];
		y[indx[i]] = c * y[indx[i]] - s * t;
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
