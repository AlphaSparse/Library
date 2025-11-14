#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t diagsv_bsr_n(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT rowA = A->rows * A->block_dim;
    TYPE diag[rowA]; //�洢�Խ�Ԫ��
    memset(diag, '\0', sizeof(TYPE) * rowA);
    ALPHA_INT bs = A->block_dim;
    
    for (ALPHA_INT ar = 0; ar < block_rowA; ++ar)
    {
        for (ALPHA_INT ai = A->row_data[ar]; ai < A->row_data[ar+1]; ++ai)
        {
            if (A->col_data[ai] == ar) //�Խǿ�
            {
                //diag[ar] = mat->values[ai];
                for(ALPHA_INT block_i = 0; block_i < bs; block_i++) //���ʿ��ڶԽ�Ԫ��
                {
                    //diag[ar*bs+block_i] = ((TYPE *)A->val_data)[ai*bs*bs + block_i*bs + block_i];
                    diag[ar*bs+block_i] = ((TYPE *)A->val_data)[ai*bs*bs + block_i*bs + block_i];
                }
            } 
        }   
    }
    TYPE tmp;
    for (ALPHA_INT r = 0; r < A->rows * A->block_dim; ++r) // y/diag
    {
        //y[r] = alpha * x[r] / diag[r];
        tmp = alpha_mul(alpha, x[r]); 
        y[r] = alpha_div(tmp, diag[r]); 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t diagsv_bsr_u(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows * A->block_dim; r++)
    {
        //y[r] = alpha * x[r];
        y[r] = alpha_mul(alpha, x[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
