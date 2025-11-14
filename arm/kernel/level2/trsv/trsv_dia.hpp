#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t trsv_dia_n_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->rows];
    ALPHA_INT main_diag_pos = 0;
    memset(diag, '\0', A->rows * sizeof(TYPE));
    for (ALPHA_INT i = 0; i < A->ndiag; i++)
    {
        if(A->dis_data[i] == 0)
        {
            main_diag_pos = i;
            for (ALPHA_INT r = 0; r < A->rows; r++)
            {
                diag[r] = ((TYPE *)A->val_data)[i * A->lval + r];
            }
            break;
        }
    }

    for (ALPHA_INT r = A->rows - 1; r >= 0; r--)
    {
        TYPE temp;
        temp = alpha_setzero(temp);
        for (ALPHA_INT ndiag = main_diag_pos + 1; ndiag < A->ndiag; ndiag++)
        {
            if (A->rows - A->dis_data[ndiag] > r)
            {
                ALPHA_INT ac = r + A->dis_data[ndiag];
                temp = alpha_madde(temp, ((TYPE *)A->val_data)[ndiag * A->lval + r], y[ac]);
                // temp += ((TYPE *)A->val_data)[ndiag * A->lval + r] * y[ac];
            }
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
alphasparseStatus_t trsv_dia_n_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    TYPE diag[A->rows];
    ALPHA_INT main_diag_pos = 0;
    memset(diag, '\0', A->rows * sizeof(TYPE));
    for (ALPHA_INT i = 0; i < A->ndiag; i++)
    {
        if(A->dis_data[i] == 0)
        {
            main_diag_pos = i;
            for (ALPHA_INT r = 0; r < A->rows; r++)
            {
                diag[r] = ((TYPE *)A->val_data)[i * A->lval + r];
            }
            break;
        }
    }

    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        TYPE temp;
        temp = alpha_setzero(temp);
        for (ALPHA_INT ndiag = 0; ndiag < main_diag_pos; ndiag++)
        {
            if (-A->dis_data[ndiag] <= r)
            {
                ALPHA_INT ac = r + A->dis_data[ndiag];
                temp = alpha_madde(temp, ((TYPE *)A->val_data)[ndiag * A->lval + r], y[ac]);
                // temp += ((TYPE *)A->val_data)[ndiag * A->lval + r] * y[ac];
            }
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
alphasparseStatus_t trsv_dia_u_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    ALPHA_INT main_diag_pos = 0;
    
    for (ALPHA_INT i = 0; i < A->ndiag; i++)
        if(A->dis_data[i] == 0)
        {
            main_diag_pos = i;
            break;
        }

    for (ALPHA_INT r = A->rows - 1; r >= 0; r--)
    {
        TYPE temp;
        temp = alpha_setzero(temp);
        for (ALPHA_INT ndiag = main_diag_pos + 1; ndiag < A->ndiag; ndiag++)
        {
            if (A->rows - A->dis_data[ndiag] > r)
            {
                ALPHA_INT ac = r + A->dis_data[ndiag];
                temp = alpha_madde(temp, ((TYPE *)A->val_data)[ndiag * A->lval + r], y[ac]);
                // temp += ((TYPE *)A->val_data)[ndiag * A->lval + r] * y[ac];
            }
        }
        TYPE t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(t, temp);
        // y[r] = alpha * x[r] - temp;
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_dia_u_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    ALPHA_INT main_diag_pos = 0;

    for (ALPHA_INT i = 0; i < A->ndiag; i++)
        if(A->dis_data[i] == 0)
        {
            main_diag_pos = i;
            break;
        }

    for (ALPHA_INT r = 0; r < A->rows; r++)
    {
        TYPE temp;
        temp = alpha_setzero(temp);
        for (ALPHA_INT ndiag = 0; ndiag < main_diag_pos; ndiag++)
        {
            if (-A->dis_data[ndiag] <= r)
            {
                ALPHA_INT ac = r + A->dis_data[ndiag];
                temp = alpha_madde(temp, ((TYPE *)A->val_data)[ndiag * A->lval + r], y[ac]);
                // temp += ((TYPE *)A->val_data)[ndiag * A->lval + r] * y[ac];
            }
        }
        TYPE t;
        t = alpha_setzero(t);
        t = alpha_mul(alpha, x[r]);
        y[r] = alpha_sub(t, temp);
        // y[r] = alpha * x[r] - temp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}