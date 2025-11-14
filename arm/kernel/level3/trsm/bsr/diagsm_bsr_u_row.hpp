#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagsm_bsr_u_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
#ifdef DEBUG
    printf("kernel diagsm_bsr_u_row called \n");
#endif
    for (ALPHA_INT r = 0; r < A->rows * A->block_dim; ++r)
    {
        for (ALPHA_INT c = 0; c < columns; ++c)
        {
            y[index2(r, c, ldy)] = alpha_mul(alpha , x[index2(r, c, ldx)]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
