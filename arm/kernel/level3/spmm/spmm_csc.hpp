#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <stdbool.h>
#include <memory.h>

template <typename TYPE>
alphasparseStatus_t spmm_csc(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    // ϡ�����A * ϡ�����B -> ϡ�����matC
    // check_return(A->cols != B->rows, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *matC = mat;
    mat->rows = A->rows;
    mat->cols = B->cols;

    ALPHA_INT m = A->rows;
    ALPHA_INT n = B->cols;
    // ��������ռ䣨��������matC��û�з���ռ�ģ�
    bool *flag = (bool *)alpha_memalign(sizeof(bool) * m, DEFAULT_ALIGNMENT);
    ALPHA_INT nnz = 0;
    for (ALPHA_INT bc = 0; bc < n; bc++)
    {
        memset(flag, '\0', sizeof(bool) * m);
        for (ALPHA_INT bi = B->col_data[bc]; bi < B->col_data[bc+1]; bi++)
        {
            ALPHA_INT ac = B->row_data[bi];
            for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac+1]; ai++)
            {
                if (!flag[A->row_data[ai]])
                {
                    nnz += 1;
                    flag[A->row_data[ai]] = true;
                }
            }
        }
    }
    alpha_free(flag);
    //printf("%d", nnz);

    ALPHA_INT *col_offset = (ALPHA_INT *)alpha_memalign(sizeof(ALPHA_INT) * (n + 1), DEFAULT_ALIGNMENT);
    mat->col_data = col_offset;
    ALPHA_INT * cols_end = col_offset + 1;
    mat->row_data = (ALPHA_INT *)alpha_memalign(nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = alpha_memalign(nnz * sizeof(TYPE), DEFAULT_ALIGNMENT);
    memset(mat->val_data, '\0', sizeof(TYPE) * nnz); 

    TYPE *values = (TYPE *)alpha_memalign(sizeof(TYPE) * m, DEFAULT_ALIGNMENT);
    bool *write_back = (bool *)alpha_memalign(sizeof(bool) * m, DEFAULT_ALIGNMENT);

    ALPHA_INT index = 0;
    mat->col_data[0] = 0;
    for (ALPHA_INT bc = 0; bc < n; bc++) //����B����
    {
        memset(values, '\0', sizeof(TYPE) * m); //matC��ÿһ����m��Ԫ��
        memset(write_back, '\0', sizeof(bool) * m);
        for (ALPHA_INT bi = B->col_data[bc]; bi < B->col_data[bc+1]; bi++) //��B�ĵ�bc��������˷�
        {
            ALPHA_INT ac = B->row_data[bi];
            TYPE bv;
            bv = ((TYPE *)B->val_data)[bi];
            for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac+1]; ai++)
            {
                ALPHA_INT ar = A->row_data[ai];
                //values[ar] += bv * ((TYPE *)A->val_data)[ai];
                TYPE av = ((TYPE *)A->val_data)[ai];
                values[ar] = alpha_madde(values[ar], bv, av);
                write_back[ar] = true;
            }
        }
        for (ALPHA_INT r = 0; r < m; r++)// ��matC�ĵ�ar�еķ���Ԫ�ر����csr
        {
            if (write_back[r])
            {
                mat->row_data[index] = r;
                //((TYPE *)mat->val_data)[index] = values[r];
                ((TYPE *)mat->val_data)[index] = values[r];
                //((TYPE *)mat->val_data)[index] = values[r];
                index += 1;
            }
        }
        cols_end[bc] = index;
    }

    alpha_free(values);
    alpha_free(write_back);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
