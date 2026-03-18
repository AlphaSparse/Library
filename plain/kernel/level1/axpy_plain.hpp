#include "alphasparse/kernel_plain.h"
#include "alphasparse/spdef.h"

template <typename IndexType = ALPHA_INT, typename ValueType>
alphasparseStatus_t axpy_plain(const IndexType nz,
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
