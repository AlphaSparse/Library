#ifndef TRANSPOSE_CONJ_CSC_HPP
#define TRANSPOSE_CONJ_CSC_HPP
#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

template <typename J>
alphasparseStatus_t transpose_conj_csc(const internal_spmat A, internal_spmat *B)
{
    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *B = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT colA = A->cols;
    mat->rows = colA;
    mat->cols = rowA;
    ALPHA_INT nnz = A->col_data[colA - 1];
    ALPHA_INT *cols_offset = (ALPHA_INT*)alpha_memalign((uint64_t)(mat->cols + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->col_data = cols_offset;
    // mat->cols_end = cols_offset + 1;
    mat->row_data = (ALPHA_INT*)alpha_memalign((uint64_t)nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = (J*)alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);
    ALPHA_INT row_counter[rowA];
    ALPHA_INT col_offset[mat->cols];
    memset(row_counter, '\0', (uint64_t)rowA * sizeof(ALPHA_INT));
    for (ALPHA_INT i = 0; i < nnz; ++i)
    {
        row_counter[A->row_data[i]] += 1;
    }
    col_offset[0] = 0;
    mat->col_data[0] = 0;
    for (ALPHA_INT i = 1; i < mat->cols; ++i)
    {
        col_offset[i] = col_offset[i - 1] + row_counter[i - 1];
        mat->col_data[i] = col_offset[i];
    }
    mat->col_data[mat->cols] = nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->col_data+1, mat->cols, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lcs = partition[tid];
        ALPHA_INT lch = partition[tid + 1];
        for (ALPHA_INT ac = 0; ac < colA; ++ac)
        {
            for (ALPHA_INT ai = A->col_data[ac]; ai < A->col_data[ac + 1]; ++ai)
            {
                ALPHA_INT bc = A->row_data[ai];
                if (bc < lcs || bc >= lch)
                    continue;
                ALPHA_INT index = col_offset[bc];
                mat->row_data[index] = ac;
                ((J*)(mat->val_data))[index] = cmp_conj(((J*)(A->val_data))[ai]);
                col_offset[bc] += 1;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#endif