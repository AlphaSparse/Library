#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t trsv_bsr_n_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT rowA = A->rows * A->block_dim;
    TYPE diag[rowA]; //存储对角元素
    memset(diag, '\0', sizeof(TYPE) * rowA);
    ALPHA_INT bs = A->block_dim;
    
    for (ALPHA_INT ar = 0; ar < block_rowA; ++ar)
    {
        for (ALPHA_INT ai = A->row_data[ar]; ai < A->row_data[ar+1]; ++ai)
        {
            if (A->col_data[ai] == ar) //对角块
            {
                //diag[ar] = ((TYPE *)A->val_data)[ai];
                for(ALPHA_INT block_i = 0; block_i < bs; block_i++) //访问块内对角元素
                {
                    //diag[ar*bs+block_i] = ((TYPE *)A->val_data)[ai*bs*bs + block_i*bs + block_i];
                    diag[ar*bs+block_i] = ((TYPE *)A->val_data)[ai*bs*bs + block_i*bs + block_i];
                }
            } 
        }   
    }
    
    TYPE temp[rowA];
    memset(temp, '\0', sizeof(TYPE)*rowA);
    for (ALPHA_INT r = block_rowA-1; r >=0 ; r--)
    {
        for (ALPHA_INT ai = A->row_data[r+1]-1; ai >= A->row_data[r]; ai--)
        {
            ALPHA_INT ac = A->col_data[ai];
            if(ac == r) //对角块
            {
                for(ALPHA_INT block_r = bs-1; block_r>=0 ; block_r--)
                {
                    for(ALPHA_INT block_c = bs-1; block_c >= 0; block_c--) //块内上三角
                    {
                        if(block_r == block_c)
                        {
                            //y[r*bs + block_r] = (alpha * x[r*bs + block_r] - temp[r*bs + block_r]) / diag[r*bs + block_r];
                            TYPE t;
                            t = alpha_mul(alpha, x[r*bs + block_r]);
                            t = alpha_sub(t, temp[r*bs + block_r]);
                            y[r*bs + block_r] = alpha_div(t, diag[r*bs + block_r]);
			                continue;
                        }
			            if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
            else if (ac > r) //上三角块
            {
                for(ALPHA_INT block_r = 0; block_r < bs; block_r++)
                {
                    for(ALPHA_INT block_c = 0; block_c < bs; block_c++)
                    {
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_n_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    //printf("trsv_d_bsr_n_lo called");
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
                    diag[ar*bs+block_i] = ((TYPE *)A->val_data)[ai*bs*bs + block_i*bs + block_i];
                }
            } 
        }   
    }
    
    TYPE temp[rowA];
    memset(temp, '\0', sizeof(TYPE)*rowA);
    for (ALPHA_INT r = 0; r < block_rowA; r++)
    {
        for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
        {
            ALPHA_INT ac = A->col_data[ai];
            if(ac == r) //�Խǿ�
            {
                for(ALPHA_INT block_r = 0; block_r < bs; block_r++)
                {
                    for(ALPHA_INT block_c = 0; block_c <= block_r; block_c++) //����������
                    {
                        if(block_c == block_r)
                        {
                            //y[r*bs + block_r] = (alpha * x[r*bs + block_r] - temp[r*bs + block_r]) / diag[r*bs + block_r];
                            TYPE t;
                            t = alpha_mul(alpha, x[r*bs + block_r]);
                            t = alpha_sub(t, temp[r*bs + block_r]);
                            y[r*bs + block_r] = alpha_div(t, diag[r*bs + block_r]);
                            continue;
                        }
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
            else if (ac < r) //�����ǿ�
            {
                for(ALPHA_INT block_r = 0; block_r < bs; block_r++)
                {
                    for(ALPHA_INT block_c = 0; block_c < bs; block_c++) //����������
                    {
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_u_hi(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT rowA = A->rows * A->block_dim;
    ALPHA_INT bs = A->block_dim;
    
    TYPE temp[rowA];
    memset(temp, '\0', sizeof(TYPE)*rowA);
    for (ALPHA_INT r = block_rowA-1; r >=0 ; r--)
    {
        for (ALPHA_INT ai = A->row_data[r+1]-1; ai >= A->row_data[r]; ai--)
        {
            ALPHA_INT ac = A->col_data[ai];
            if(ac == r) //�Խǿ�
            {
                for(ALPHA_INT block_r = bs-1; block_r>=0 ; block_r--)
                {
                    for(ALPHA_INT block_c = bs-1; block_c >= 0; block_c--) //����������
                    {
			            if(block_r == block_c)
                        {
                            //y[r*bs + block_r] = (alpha * x[r*bs + block_r] - temp[r*bs + block_r]);
                            TYPE t;
                            t = alpha_mul(alpha, x[r*bs + block_r]);
                            y[r*bs + block_r] = alpha_sub(t, temp[r*bs + block_r]);
                            continue;
                        }
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
            else if (ac > r) //�����ǿ�
            {
                for(ALPHA_INT block_r = 0; block_r < bs; block_r++)
                {
                    for(ALPHA_INT block_c = 0; block_c < bs; block_c++)
                    {
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t trsv_bsr_u_lo(const TYPE alpha, const internal_spmat A, const TYPE *x, TYPE *y)
{
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT rowA = A->rows * A->block_dim;
    ALPHA_INT bs = A->block_dim;

    
    TYPE temp[rowA];
    memset(temp, '\0', sizeof(TYPE)*rowA);
    for (ALPHA_INT r = 0; r < block_rowA; r++)
    {
        for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ai++)
        {
            ALPHA_INT ac = A->col_data[ai];
            if(ac == r) //�Խǿ�
            {
                for(ALPHA_INT block_r = 0; block_r < bs; block_r++)
                {
                    for(ALPHA_INT block_c = 0; block_c <= block_r; block_c++) //����������
                    {
			            if(block_r == block_c)
                        {
                            //y[r*bs + block_r] = (alpha * x[r*bs + block_r] - temp[r*bs + block_r]);
                            TYPE t;
                            t = alpha_mul(alpha, x[r*bs + block_r]);
                            y[r*bs + block_r] = alpha_sub(t, temp[r*bs + block_r]);
                            continue;
                        }
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
            else if (ac < r) //�����ǿ�
            {
                for(ALPHA_INT block_r = 0; block_r < bs; block_r++)
                {
                    for(ALPHA_INT block_c = 0; block_c < bs; block_c++)
                    {
                        if(A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR)
                        {
                            // A row major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_r*bs + block_c], y[ac*bs + block_c]);
                        }
                        else if(A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                        {
                            // A column major
                            //temp[r*bs + block_r] += ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r] * y[ac*bs + block_c];
                            temp[r*bs + block_r] = alpha_madde(temp[r*bs + block_r], ((TYPE *)A->val_data)[ai*bs*bs + block_c*bs + block_r], y[ac*bs + block_c]);
                        }
                        else
                        {
                            return ALPHA_SPARSE_STATUS_INVALID_VALUE;
                        }
                    }
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
