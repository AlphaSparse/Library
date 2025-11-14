#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t diagsv_csc_n(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->cols];

    memset(diag, '\0', A->cols * sizeof(TYPE));

    for (ALPHA_INT c = 0; c < A->cols; c++)// ��ȡ�Խ���
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
    TYPE tmp;
    for (ALPHA_INT r = 0; r < A->rows; ++r) // y/diag
    {
        //y[r] = alpha * x[r] / diag[r];
        tmp = alpha_mul(alpha, x[r]); 
        y[r] = alpha_div(tmp, diag[r]); 
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t diagsv_csc_u(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++) //����unit���Խ���Ԫ�ض�����ȡ��
    {
        //y[r] = alpha * x[r];
        y[r] = alpha_mul(alpha, x[r]);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
