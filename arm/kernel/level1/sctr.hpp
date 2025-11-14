#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
alphasparseStatus_t sctr(const IndexType nz,
	  const ValueType *x,
	  const IndexType *indx,
	  ValueType *y)
{
	for (IndexType i = 0; i < nz; ++i)
	{
		y[indx[i]] = x[i];
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
