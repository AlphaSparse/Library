#include "alphasparse/kernel.h"
#include "alphasparse/spdef.h"
#include "alphasparse/compute.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
alphasparseStatus_t axpy(const IndexType nz,
                           const ValueType a,
                           const ValueType* x,
                           const IndexType* indx,
                           ValueType* y)
{
    for (IndexType i = 0; i < nz; ++i)
    {
        y[indx[i]] = alpha_madd(y[indx[i]], a, x[i]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
