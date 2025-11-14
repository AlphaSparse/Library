#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>
#include "alphasparse/compute.h"

//plain
template <typename I, typename J, typename W>
alphasparseStatus_t trsv_csr_n_hi(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* rows_start, 
                          const W* rows_end, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    J diag[rows];
    memset(diag, '\0', rows * sizeof(J));
    for (I r = 0; r < rows; r++)
    {
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            I ac = col_indx[ai];
            if (ac == r)
            {
                diag[r] = values[ai];
            }
        }
    }
    for (I r = rows - 1; r >= 0; r--)
    {
        J temp;
        temp = alpha_setzero(temp);
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            I ac = col_indx[ai];
            if (ac > r)
            {
                temp = alpha_madde(temp, values[ai], y[ac]);
            }
        }
        J t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[r]);
        t = alpha_sub(t, temp);
        y[r] = alpha_div(t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t trsv_csr_u_hi(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* rows_start, 
                          const W* rows_end, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    for (I r = rows - 1; r >= 0; r--)
    {
        J temp;
        temp = alpha_setzero(temp);
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            I ac = col_indx[ai];
            if (ac > r)
            {
                temp = alpha_madde(temp, values[ai], y[ac]);
            }
        }
        y[r] = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(y[r], temp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t trsv_csr_n_lo(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* rows_start, 
                          const W* rows_end, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    J diag[rows];
    memset(diag, '\0', rows * sizeof(J));
    for (I r = 0; r < rows; r++)
    {
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            I ac = col_indx[ai];
            if (ac == r)
            {
                diag[r] = values[ai];
            }
        }
    }
    for (I r = 0; r < rows; r++)
    {
        J temp;
        temp = alpha_setzero(temp);
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            I ac = col_indx[ai];
            if (ac < r)
            {
                temp = alpha_madde(temp, values[ai], y[ac]);
            }
        }
        J t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[r]);
        t = alpha_sub(t, temp);
        y[r] = alpha_div(t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t trsv_csr_u_lo(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* rows_start, 
                          const W* rows_end, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    for (I r = 0; r < rows; r++)
    {
        J temp;
        temp = alpha_setzero(temp);
        for (I ai = rows_start[r]; ai < rows_end[r]; ai++)
        {
            I ac = col_indx[ai];
            if (ac < r)
            {
                temp = alpha_madde(temp, values[ai], y[ac]);
            }
        }
        y[r] = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(y[r], temp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

//trans
// template <typename I, typename J, typename W>
// alphasparseStatus_t trsv_csr_n_hi_trans(const J alpha, 
//                           const I rows, 
//                           const I cols,
//                           const W* rows_start, 
//                           const W* rows_end, 
//                           const W* col_indx,
//                           const J *values,    
//                           const J *x, 
//                           J *y)
// {
//     ALPHA_SPMAT_CSR *transposed_mat;
//     transpose_csr(A, &transposed_mat);
//     alphasparse_status_t status = trsv_csr_n_lo(alpha, transposed_mat, x, y);
//     destroy_csr(transposed_mat);
//     return status;
// }
