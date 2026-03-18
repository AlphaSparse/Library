#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include <type_traits>

template <typename I , typename J, typename W>
alphasparseStatus_t gemv_coo_plain(const J alpha,
                                const I rows,
                                const I cols,
                                const I nnz,
                                const W* row_indx, 
                                const W* col_indx,
                                const J *values,
                                const J *x,
                                const J beta,
                                J *y)
{
	for (I i = 0; i < rows; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}
    for (I i = 0; i < nnz; i++)
    {
        I r = row_indx[i];
		I c = col_indx[i];
		J v;
		v = alpha_mul(values[i], x[c]);
		y[r] = alpha_madd(y[r], alpha, v);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t gemv_coo_trans_plain(const J alpha,
                                const I rows,
                                const I cols,
                                const I nnz,
                                const W* row_indx, 
                                const W* col_indx,
                                const J *values,
                                const J *x,
                                const J beta,
                                J *y)
{
	for (I i = 0; i < cols; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}
    for (I i = 0; i < nnz; i++)
    {
        I r = row_indx[i];
		I c = col_indx[i];
		J v;
		v = alpha_mul(values[i], x[r]);
		y[c] = alpha_madd(y[c], alpha, v);
        
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
template <typename I, typename J, typename W>
alphasparseStatus_t gemv_coo_conj_plain(const J alpha,
                                const I rows,
                                const I cols,
                                const I nnz,
                                const W* row_indx, 
                                const W* col_indx,
                                const J *values,
                                const J *x,
                                const J beta,
                                J *y)
{
	for (I i = 0; i < cols; i++)
	{
		y[i] = alpha_mul(y[i], beta);
	}
    for (I i = 0; i < nnz; i++)
    {
        I r = row_indx[i];
		I c = col_indx[i];
		J v;
		v = cmp_conj(values[i]);
		v = alpha_mul(v, x[r]);
		y[c] = alpha_madd(y[c], alpha, v);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

