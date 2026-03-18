#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include <type_traits>

template <typename I, typename J, typename W>
alphasparseStatus_t gemv_csr_plain(const J alpha,
                                const I rows, const I cols,
                                const W* rows_start, 
                                const W* rows_end, 
                                const W* col_indx,
                                const J *values,
                                const J *x,
                                const J beta,
                                J *y)
{
    for (I r = 0; r < rows; r++)
    {
        y[r] = alpha_mul(y[r], beta);
        J tmp;
        tmp = alpha_setzero(tmp);
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            tmp = alpha_madd(tmp, values[ai], x[col_indx[ai]]);
        }
        y[r] = alpha_madd(y[r], alpha, tmp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t gemv_csr_trans_plain(const J alpha,
                                const I rows, const I cols,
                                const W* rows_start, 
                                const W* rows_end, 
                                const W* col_indx,
                                const J *values,
                                const J *x,
                                const J beta,
                                J *y)
{
    for (I j = 0; j < cols; ++j)
    {
        y[j] = alpha_mul(y[j], beta);
    }
    for (I i = 0; i < rows; i++)
    {
        for (I ai = rows_start[i]; ai < rows_end[i]; ai++)
        {
            J val;
            val = alpha_mul(alpha, values[ai]);
            y[col_indx[ai]] = alpha_madd(y[col_indx[ai]], val, x[i]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


template <typename I, typename J, typename W>
alphasparseStatus_t gemv_csr_conj_plain(const J alpha,
                                const I rows, const I cols,
                                const W* rows_start, 
                                const W* rows_end, 
                                const W* col_indx,
                                const J *values,
                                const J *x,
                                const J beta,
                                J *y)
{
    for (I j = 0; j < cols; ++j)
    {
        y[j] = alpha_mul(y[j], beta);
    }
    for (I i = 0; i < rows; i++)
    {
        for (I ai = rows_start[i]; ai < rows_end[i]; ai++)
        {
            J val = values[ai];
            val = cmp_conj(val);
            val = alpha_mul(alpha, val);
            y[col_indx[ai]] = alpha_madd(y[col_indx[ai]], val, x[i]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}