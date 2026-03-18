#ifndef TRANSPOSE_BSR_HPP
#define TRANSPOSE_BSR_HPP
#include "alphasparse/format.h"
#include <stdlib.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>

template <typename J>
alphasparseStatus_t transpose_bsr(const internal_spmat A, internal_spmat *B)
{
    internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(_internal_spmat));
    *B = mat;
    ALPHA_INT block_dim = A->block_dim;
    ALPHA_INT rowA = A->rows * block_dim;
    ALPHA_INT colA = A->cols * block_dim;
    ALPHA_INT block_rowA = A->rows;
    ALPHA_INT block_colA = A->cols;
    mat->rows = A->cols;
    mat->cols = A->rows;
    mat->block_dim = block_dim;
    mat->block_layout = A->block_layout;
    ALPHA_INT block_nnz = A->nnz;
    ALPHA_INT *rows_offset = (ALPHA_INT*)alpha_memalign((uint64_t)(block_colA + 1) * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->row_data = rows_offset;
    // mat->rows_end = rows_offset + 1;
    mat->col_data = (ALPHA_INT*)alpha_memalign((uint64_t)block_nnz * sizeof(ALPHA_INT), DEFAULT_ALIGNMENT);
    mat->val_data = (J*)alpha_memalign((uint64_t)block_nnz * block_dim * block_dim * sizeof(J), DEFAULT_ALIGNMENT);
    ALPHA_INT col_counter[block_colA];
    ALPHA_INT row_offset[block_colA];
    memset(col_counter, '\0', (uint64_t)block_colA * sizeof(ALPHA_INT));
    for (ALPHA_INT i = 0; i < block_nnz; ++i)
    {
        col_counter[A->col_data[i]] += 1;
    }
    row_offset[0] = 0;
    mat->row_data[0] = 0;
    for (ALPHA_INT i = 1; i < block_colA; ++i)
    {
        row_offset[i] = row_offset[i - 1] + col_counter[i - 1];
        mat->row_data[i] = row_offset[i];
    }
    mat->row_data[block_colA] = block_nnz;
    for (ALPHA_INT r = 0; r < block_rowA; ++r)
    {
        for (ALPHA_INT ai = A->row_data[r]; ai < A->row_data[r+1]; ++ai)
        {
            ALPHA_INT ac = A->col_data[ai];
            ALPHA_INT index = row_offset[ac];
            mat->col_data[index] = r;
            const J* A_values = (J*)(A->val_data) + ai * block_dim * block_dim;
            J* B_values = (J*)(mat->val_data) + index * block_dim * block_dim;
            for(ALPHA_INT br = 0;br < block_dim;++br){
                for(ALPHA_INT bc = 0;bc < block_dim;++bc){
                    B_values[index2(bc,br,block_dim)] = A_values[index2(br,bc,block_dim)];       
                }
            }
            row_offset[ac] += 1;
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
#endif