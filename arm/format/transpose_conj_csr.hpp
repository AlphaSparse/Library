#ifndef TRANSPOSE_CONJ_CSR_HPP
#define TRANSPOSE_CONJ_CSR_HPP
#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <alphasparse/compute.h>
#include <memory.h>

template <typename J>
alphasparseStatus_t transpose_conj_csr(const internal_spmat A, internal_spmat *B)
{
    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *B = mat;
    ALPHA_INT rowA = A->rows;
    ALPHA_INT colA = A->cols;
    mat->rows = colA;
    mat->cols = rowA;
    ALPHA_INT nnz = A->row_data[rowA];
    ALPHA_INT *rows_offset = (ALPHA_INT*)alpha_memalign((uint64_t)(mat->rows + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->row_data = rows_offset;
    // mat->rows_end = rows_offset + 1;
    mat->col_data = (ALPHA_INT*)alpha_memalign((uint64_t)nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = (J*)alpha_memalign((uint64_t)nnz * sizeof(J), DEFAULT_ALIGNMENT);
    ALPHA_INT col_counter[colA];
    ALPHA_INT row_offset[colA];
    memset(col_counter, '\0', (uint64_t)colA * sizeof(ALPHA_INT));
    for (ALPHA_INT i = 0; i < nnz; ++i)
    {
        col_counter[A->col_data[i]] += 1;
    }
    row_offset[0] = 0;
    mat->row_data[0] = 0;
    for (ALPHA_INT i = 1; i < colA; ++i)
    {
        row_offset[i] = row_offset[i - 1] + col_counter[i - 1];
        mat->row_data[i] = row_offset[i];
    }
    mat->row_data[colA] = nnz;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->row_data + 1, mat->rows, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT lrs = partition[tid];
        ALPHA_INT lrh = partition[tid + 1];
        for (ALPHA_INT r = 0; r < rowA; ++r)
        {
            for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ++ai)
            {
                ALPHA_INT ac = A->col_data[ai];
                if (ac < lrs || ac >= lrh)
                    continue;
                ALPHA_INT index = row_offset[ac];
                mat->col_data[index] = r;
                ((J*)(mat->val_data))[index] = cmp_conj(((J*)(A->val_data))[ai]);
                row_offset[ac] += 1;
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#endif