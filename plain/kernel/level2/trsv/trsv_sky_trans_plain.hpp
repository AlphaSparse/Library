#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t trsv_sky_n_hi_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    alphasparseStatus_t status = trsv_sky_n_lo_plain(alpha, A, x, y);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_sky_n_lo_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    alphasparseStatus_t status = trsv_sky_n_hi_plain(alpha, A, x, y);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_sky_u_hi_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    alphasparseStatus_t status = trsv_sky_u_lo_plain(alpha, A, x, y);
    return status;
}

template <typename TYPE>
alphasparseStatus_t trsv_sky_u_lo_trans_plain(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    alphasparseStatus_t status = trsv_sky_u_hi_plain(alpha, A, x, y);
    return status;
}