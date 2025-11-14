#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include <memory.h>
#include "alphasparse/compute.h"

template <typename I, typename J, typename W>
alphasparseStatus_t trsv_coo_n_hi(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* row_indx, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    J diag[rows];
    memset(diag, '\0', rows * sizeof(J));

    for (I r = 0; r < nnz; r++)
    {
        if(row_indx[r] == col_indx[r])
        {
            diag[row_indx[r]] = values[r];
        }
    }

    for (I r = rows - 1; r >= 0; r--)
    {
        J temp;
        temp = alpha_setzero(temp);

        for (I cr = 0; cr < nnz; cr++)
        {
            int row = row_indx[cr];
            int col = col_indx[cr];
            if(row == r && col > r)
            {
                temp = alpha_madde(temp, values[cr], y[col]);
            }
        }
        J t;
        t = alpha_mul(alpha, x[r]);
        t = alpha_sub(t, temp);
        y[r] = alpha_div(t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t trsv_coo_u_hi(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* row_indx, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    for (I r = rows - 1; r >= 0; r--)
    {
        J temp;
        temp = alpha_setzero(temp);

        for (I cr = 0; cr < nnz; cr++)
        {
            int row = row_indx[cr];
            int col = col_indx[cr];
            if(row == r && col > r)
            {
                temp = alpha_madde(temp, values[cr], y[col]);
            }
        }
        J t;
        t = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(t, temp); 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


template <typename I, typename J, typename W>
alphasparseStatus_t trsv_coo_n_lo(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* row_indx, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y){
    J diag[rows];
    memset(diag, '\0', rows * sizeof(J));

    for (I r = 0; r < nnz; r++)
    {
        if(row_indx[r] == col_indx[r])
        {
            diag[row_indx[r]] = values[r];
        }
    }

    for (I r = 0; r < rows; r++)
    {
        J temp;
        temp = alpha_setzero(temp);
        for (I cr = 0; cr < nnz; cr++)
        {
            int row = row_indx[cr];
            int col = col_indx[cr];
            if(row == r && col < r)
            {
                temp = alpha_madde(temp, values[cr], y[col]);
            }
        }
        J t;
        t = alpha_mul(alpha, x[r]);
        t = alpha_sub(t, temp);
        y[r] = alpha_div(t, diag[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t trsv_coo_u_lo(const J alpha, 
                          const I rows, 
                          const I cols,
                          const I nnz,
                          const W* row_indx, 
                          const W* col_indx,
                          const J *values,    
                          const J *x, 
                          J *y)
{
    for (I r = 0; r < rows; r++)
    {
        J temp;
        temp = alpha_setzero(temp);

        for (I cr = 0; cr < nnz; cr++)
        {
            int row = row_indx[cr];
            int col = col_indx[cr];
            if(row == r && col < r)
            {
                temp = alpha_madde(temp, values[cr], y[col]);
            }
        }
        J t;
        t = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(t, temp);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
