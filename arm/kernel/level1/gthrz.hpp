#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
alphasparseStatus_t gthrz(const IndexType nz,
	  ValueType *y,
	  ValueType *x,
	  const IndexType *indx)
{
	for (IndexType i = 0; i < nz; ++i)
	{
		x[i] = y[indx[i]];
		alpha_setzero(y[indx[i]]);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
