#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/transpose_conj_coo.hpp"
#include "format/destroy_coo.hpp"

#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagsm_coo_n_row(const J alpha, const internal_spmat A, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, J *y, const ALPHA_INT ldy)
{
    J diag[A->rows];

    memset(diag, '\0', A->rows * sizeof(J));

    for (ALPHA_INT r = 0; r < A->nnz; r++)
    {
        if(A->row_data[r] == A->col_data[r])
        {
            // diag[A->row_data[r]].real = ((J*)A->val_data)[r].real;
            // diag[A->row_data[r]].imag = ((J*)A->val_data)[r].imag;
            diag[A->row_data[r]] = ((J*)A->val_data)[r];
        }
    }
    
    for (ALPHA_INT r = 0; r < A->rows; ++r)
    {
        for (ALPHA_INT c = 0; c < columns; ++c)
        {
            J t;
            t = alpha_mul(alpha, x[index2(r, c, ldx)]);
            y[index2(r, c, ldy)] = alpha_div(t, diag[r]);
            // y[index2(r, c, ldy)] = alpha * x[index2(r, c, ldx)] / diag[r];
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
