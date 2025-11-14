#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagsm_csr_n_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    J diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(J));

    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
        {
            ALPHA_INT ac = A->col_data[ai];
            if (ac == r)
            {
                diag[r] = ((J*)A->val_data)[ai];
            }
        }
    }
    
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        for (ALPHA_INT c = 0; c < columns; ++c)
        {
            J t;
            t = alpha_setzero(t);
            t = alpha_mul(alpha, x[index2(r, c, ldx)]);
            y[index2(r, c, ldy)] = alpha_div(t, diag[r]);
            // y[index2(r, c, ldy)] = alpha * x[index2(r, c, ldx)] / diag[r];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
