#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_csr.hpp"
#include "format/transpose_conj_csr.hpp"
#include "format/destroy_csr.hpp"
#include "alphasparse/util.h"

template <typename TYPE>
alphasparseStatus_t add_csr(const internal_spmat A, const TYPE alpha, const internal_spmat B, internal_spmat *matC)
{

    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *matC = mat;

    ALPHA_INT rowA = A->rows;
    ALPHA_INT rowB = B->rows;
    ALPHA_INT colA = A->cols;
    ALPHA_INT colB = B->cols;

    // check_return(rowA != rowB, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    // check_return(colA != colA, ALPHA_SPARSE_STATUS_INVALID_VALUE);

    ALPHA_INT *rows_offset = (ALPHA_INT *)alpha_memalign((rowA + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->rows = rowA;
    mat->cols = colB;
    mat->row_data = rows_offset;
    ALPHA_INT * rows_end = rows_offset + 1;

    // 计算需要结果需要的空间
    size_t count = 0;
    for (ALPHA_INT r = 0; r < rowA; ++r)
    {
        ALPHA_INT ai = A->row_data[r];
        ALPHA_INT rea = A->row_data[r+1];
        ALPHA_INT bi = B->row_data[r];
        ALPHA_INT reb = B->row_data[r+1];
        while (ai < rea && bi < reb)
        {
            ALPHA_INT ca = A->col_data[ai];
            ALPHA_INT cb = B->col_data[bi];
            if (ca < cb)
            {
                ai += 1;
            }
            else if (cb < ca)
            {
                bi += 1;
            }
            else
            {
                ai += 1;
                bi += 1;
            }
            count += 1;
        }
        if (ai == rea)
        {
            count += reb - bi;
        }
        else
        {
            count += rea - ai;
        }
    }
    mat->col_data = (ALPHA_INT*)alpha_memalign(count * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = alpha_memalign(count * sizeof(TYPE), DEFAULT_ALIGNMENT);

    // add
    size_t index = 0;
    mat->row_data[0] = 0;
    for (ALPHA_INT r = 0; r < rowA; ++r)
    {
        ALPHA_INT ai = A->row_data[r];
        ALPHA_INT rea = A->row_data[r+1];
        ALPHA_INT bi = B->row_data[r];
        ALPHA_INT reb = B->row_data[r+1];
        while (ai < rea && bi < reb)
        {
            ALPHA_INT ca = A->col_data[ai];
            ALPHA_INT cb = B->col_data[bi];
            if (ca < cb)
            {
                mat->col_data[index] = ca;
                ((TYPE *)mat->val_data)[index] = alpha_mul(((TYPE *)A->val_data)[ai], alpha);
                ai += 1;
            }
            else if (cb < ca)
            {
                mat->col_data[index] = cb;
                ((TYPE *)mat->val_data)[index] = ((TYPE *)B->val_data)[bi];
                bi += 1;
            }
            else
            {
                mat->col_data[index] = ca;
                ((TYPE *)mat->val_data)[index] = alpha_madd(((TYPE *)A->val_data)[ai], alpha, ((TYPE *)B->val_data)[bi]);
                ai += 1;
                bi += 1;
            }
            index += 1;
        }
        if (ai == rea)
        {
            for (; bi < reb; ++bi, ++index)
            {
                mat->col_data[index] = B->col_data[bi];
                ((TYPE *)mat->val_data)[index] = ((TYPE *)B->val_data)[bi];
            }
        }
        else
        {
            for (; ai < rea; ++ai, ++index)
            {
                mat->col_data[index] = A->col_data[ai];
                ((TYPE *)mat->val_data)[index] = alpha_mul(((TYPE *)A->val_data)[ai], alpha);
            }
        }
        rows_end[r] = index;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
