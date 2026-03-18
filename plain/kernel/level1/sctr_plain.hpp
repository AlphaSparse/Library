#include "alphasparse/kernel_plain.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
alphasparseStatus_t sctr_plain(const IndexType nz,
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
