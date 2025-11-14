#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t trsv_sky_n_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->cols];
    memset(diag, '\0', A->rows * sizeof(TYPE));
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
        diag[r - 1] = ((TYPE *)A->val_data)[indx];
    }

    for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
    {
        TYPE temp;
        temp = alpha_setzero(temp);
        for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
        {
            ALPHA_INT start = A->pointers[ic];
            ALPHA_INT end   = A->pointers[ic + 1];
            ALPHA_INT eles_num = ic - c;
            if(end - eles_num - 1 >= start)
                temp = alpha_madde(temp, ((TYPE *)A->val_data)[end - eles_num - 1], y[ic]);
                // temp += ((TYPE *)A->val_data)[end - eles_num - 1] * y[ic];
        }

        TYPE t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[c]);
        t = alpha_sub(t, temp);
        y[c] = alpha_div(t, diag[c]);
        // y[c] = (alpha * x[c] - temp) / diag[c];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_sky_n_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->cols];
    memset(diag, '\0', A->rows * sizeof(TYPE));
    for (ALPHA_INT r = 1; r < A->rows + 1; r++)
    {
        const ALPHA_INT indx = A->pointers[r] - 1;
        diag[r - 1] = ((TYPE *)A->val_data)[indx];
    }

    for (ALPHA_INT r = 0; r <A->rows; r++)
    {
        TYPE temp;
        temp = alpha_setzero(temp);

        ALPHA_INT start = A->pointers[r];
        ALPHA_INT end   = A->pointers[r + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;
        for (ALPHA_INT ai = start; ai < end - 1; ++ai)
        {
            ALPHA_INT c = r - eles_num + idx;
            temp = alpha_madde(temp, ((TYPE *)A->val_data)[ai], y[c]);
            idx ++;
        }     

        TYPE t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[r]);
        t = alpha_sub(t, temp);
        y[r] = alpha_div(t, diag[r]);
        // y[r] = (alpha * x[r] - temp) / diag[r];
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_sky_u_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT c = A->cols - 1; c >= 0; c--)
    {
        TYPE temp;
        temp = alpha_setzero(temp);
        for (ALPHA_INT ic = A->cols - 1; ic > c; ic--)
        {
            ALPHA_INT start = A->pointers[ic];
            ALPHA_INT end   = A->pointers[ic + 1];
            ALPHA_INT eles_num = ic - c;
            if(end - eles_num - 1 >= start)
                temp = alpha_madde(temp, ((TYPE *)A->val_data)[end - eles_num - 1], y[ic]);
                // temp += ((TYPE *)A->val_data)[end - eles_num - 1] * y[ic];
        }
        
        TYPE t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[c]);
        y[c] = alpha_sub(t, temp);
        // y[c] = alpha * x[c] - temp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_sky_u_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        TYPE temp;
        temp = alpha_setzero(temp);

        ALPHA_INT start = A->pointers[r];
        ALPHA_INT end   = A->pointers[r + 1];
        ALPHA_INT idx = 1;
        ALPHA_INT eles_num = end - start;
        for (ALPHA_INT ai = start; ai < end - 1; ++ai)
        {
            ALPHA_INT c = r - eles_num + idx;
            temp = alpha_madde(temp, ((TYPE *)A->val_data)[ai], y[c]);
            idx ++;
        }     

        TYPE t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(t, temp);
        // y[r] = alpha * x[r] - temp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}