#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_coo.hpp"
#include "format/destroy_coo.hpp"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t hermm_coo_n_lo_row_trans(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    //TODO faster without invoking transposed 
    internal_spmat transposed_mat;
    transpose_coo<J>(mat, &transposed_mat);
    int nnz = transposed_mat->nnz;

    alphasparseStatus_t status = hermm_coo_n_hi_row(alpha,
		                                                    transposed_mat,
                                                            x,
                                                            columns,
                                                            ldx,
                                                            beta,
                                                            y,
                                                            ldy);
    destroy_coo(transposed_mat);
    return status;

}