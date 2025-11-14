#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t trsv_csc_n_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->cols]; 
    memset(diag, '\0', A->cols * sizeof(TYPE));
    for (ALPHA_INT c = 0; c < A->cols; c++) 
    {
        for (ALPHA_INT ai = A->col_data[c]; ai < A->col_data[c+1]; ai++) 
        {
            ALPHA_INT ar = A->row_data[ai]; 
            if (ar == c)
            {
                //diag[c] = ((TYPE *)A->val_data)[ai];
                diag[c] = ((TYPE *)A->val_data)[ai];
            }
        }
    }
    
    //memset(alphax, '\0', A->cols * sizeof(TYPE));
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        //alphax[c] = alpha * x[c];  
        //y[c] = alpha * x[c];
        y[c] = alpha_mul(alpha, x[c]);
    }
    for (ALPHA_INT ac = A->cols - 1; ac >= 0; ac--) 
    {
        y[ac] = alpha_div(y[ac], diag[ac]);
        //y[ac] = y[ac] / diag[ac];   
	    for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac+1]; ai++)
        {
            ALPHA_INT ar = A->row_data[ai];
            //TYPE val = ((TYPE *)A->val_data)[ai];
            TYPE val;
            val = ((TYPE *)A->val_data)[ai];
            if (ac > ar) 
            {
                //alphax[ar] -= val * x[ac];
                //y[ar] -= val * y[ac];
                TYPE t;
                t = alpha_mul(val, y[ac]);
                y[ar] = alpha_sub(y[ar], t);
            }
        }
        //y[ac] = y[ac]/diag[ac];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_n_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->cols]; 
    memset(diag, '\0', A->cols * sizeof(TYPE));
    for (ALPHA_INT c = 0; c < A->cols; c++) //????A??????
    {
        for (ALPHA_INT ai = A->col_data[c]; ai < A->col_data[c+1]; ai++) //??????c???��?????????value??row_indx??��??
        {
            ALPHA_INT ar = A->row_data[ai]; //A???c?��???ai??????????????��?
            if (ar == c) //?��??????
            {
                //diag[c] = ((TYPE *)A->val_data)[ai];
                diag[c] = ((TYPE *)A->val_data)[ai];
            }
        }
    }
    
    //TYPE alphax[A->cols]; //???��alph*x???????????????ac??y???????????????????????��
    //memset(alphax, '\0', A->cols * sizeof(TYPE));
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        //alphax[c] = alpha * x[c];  //??alphax?????
        //y[c] = alpha * x[c];
        y[c] = alpha_mul(alpha, x[c]);
    }
    for (ALPHA_INT ac = 0; ac < A->cols; ac++) //????A??????
    {
        y[ac] = alpha_div(y[ac], diag[ac]);
        //y[ac] = y[ac] / diag[ac];   
        for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac+1]; ai++) //????A??ac???��????????values??row_indx??��??
        {
            ALPHA_INT ar = A->row_data[ai];
            //TYPE val = ((TYPE *)A->val_data)[ai];
            TYPE val;
            val = ((TYPE *)A->val_data)[ai];
            if (ac < ar) // ?????????????????=
            {
                //alphax[ar] -= val * y[ac];
                //y[ar] -= val * y[ac];
                TYPE t;
                t = alpha_mul(val, y[ac]);
                y[ar] = alpha_sub(y[ar], t);
            }
        }
        //y[ac] = alphax[ac]/diag[ac];
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_u_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    //TYPE alphax[A->cols]; //???��alph*x???????????????ac??y???????????????????????��
    //memset(alphax, '\0', A->cols * sizeof(TYPE));
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        //alphax[c] = alpha * x[c];  //??alphax?????
        //y[c] = alpha *x[c];
        y[c] = alpha_mul(alpha, x[c]);
    }
    for (ALPHA_INT ac = A->cols - 1; ac >= 0; ac--) //????A??????
    {
        for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac+1]; ai++) //????A??ac???��????????values??row_indx??��??
        {
            ALPHA_INT ar = A->row_data[ai];
            //TYPE val = ((TYPE *)A->val_data)[ai];
            TYPE val;
            val = ((TYPE *)A->val_data)[ai];
            if (ac > ar) // ?????????????????=
            {
                //alphax[ar] -= val * y[ac];
                //y[ar] -= val * y[ac];
                TYPE t;
                t = alpha_mul(val, y[ac]);
                y[ar] = alpha_sub(y[ar], t);
            }
        }
        //y[ac] = alphax[ac];
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_csc_u_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    //TYPE alphax[A->cols]; //???��alph*x???????????????ac??y???????????????????????��
    //memset(alphax, '\0', A->cols * sizeof(TYPE));
    for (ALPHA_INT c = 0; c < A->cols; c++)
    {
        //y[c] = alpha * x[c];  //??alphax?????
        y[c] = alpha_mul(alpha, x[c]);
    }
    for (ALPHA_INT ac = 0; ac < A->cols; ac++) //????A??????
    {
        for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac+1]; ai++) //????A??ac???��????????values??row_indx??��??
        {
            ALPHA_INT ar = A->row_data[ai];
            //TYPE val = ((TYPE *)A->val_data)[ai];
            TYPE val;
            val = ((TYPE *)A->val_data)[ai];
            if (ac < ar)
            {
                //y[ar] -= val * y[ac];
                TYPE t;
                t = alpha_mul(val, y[ac]);
                y[ar] = alpha_sub(y[ar], t);
            }
        }
        //y[ac] = alphax[ac];
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}