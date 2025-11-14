#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t trsm_sky_n_hi_row_trans(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    alphasparseStatus_t status = trsm_sky_n_lo_row(alpha, A, x, columns, ldx, y, ldy);
    return status;
}
