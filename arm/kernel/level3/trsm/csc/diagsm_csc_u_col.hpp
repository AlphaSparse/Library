#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csc.hpp"
#include "format/transpose_conj_csc.hpp"
#include "format/destroy_csc.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagsm_csc_u_col(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
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
