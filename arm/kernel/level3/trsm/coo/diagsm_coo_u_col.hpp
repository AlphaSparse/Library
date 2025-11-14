#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagsm_coo_u_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    for (ALPHA_INT c = 0; c < columns; ++c)
    {
        for (ALPHA_INT r = 0; r < A->rows; ++r)
        {
            y[index2(c, r, ldy)] = alpha_mul(alpha, x[index2(c, r, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
