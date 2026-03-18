#include <alphasparse/opt.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include "alphasparse/format.h"
#include "alphasparse/util.h"
#include "alphasparse/util/prefix_sum.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/util/partition.h"
#include <alphasparse/opt.h>
#include "convert_csr_coo.hpp"


template <typename I, typename J, typename T>
alphasparseStatus_t convert_csr5_coo(const T *source, T **dest) {
    // Step 1: 使用已有的 COO -> CSR 转换
    T *csr_mat = nullptr;
    alphasparseStatus_t status = convert_csr_coo<I, J, T>(source, &csr_mat);
    if (status != ALPHA_SPARSE_STATUS_SUCCESS) return status;

    // Step 2: 分配并初始化 CSR5 目标矩阵
    T *mat = (T *)alpha_malloc(sizeof(T));
    if (!mat) return ALPHA_SPARSE_STATUS_ALLOC_FAILED;
    *dest = mat;

    // Step 3: 基本信息拷贝
    mat->rows = csr_mat->rows;
    mat->cols = csr_mat->cols;
    mat->nnz = csr_mat->nnz;
    mat->idx_base = csr_mat->idx_base;

    mat->row_data = csr_mat->row_data;
    mat->col_data = csr_mat->col_data;
    mat->val_data = csr_mat->val_data;

    // Step 4: CSR5 特有信息初始化
    const I SIGMA = 16; // 每个 tile 最大行数，可调
    mat->csr5_sigma = SIGMA;

    I m = mat->rows;
    I nnz = mat->nnz;
    I *row_ptr = mat->row_data;

    // 计算 tile 总数
    I num_tiles = (m + SIGMA - 1) / SIGMA;
    mat->csr5_num_tiles = num_tiles;

    // 分配 CSR5 结构
    mat->tile_ptr = (I *)alpha_memalign((num_tiles + 1) * sizeof(I), DEFAULT_ALIGNMENT);
    mat->tile_desc = (I *)alpha_memalign(num_tiles * sizeof(I), DEFAULT_ALIGNMENT);
    mat->tile_row_offset = (I *)alpha_memalign(num_tiles * sizeof(I), DEFAULT_ALIGNMENT);
    mat->csr_offset = (I *)alpha_memalign(num_tiles * sizeof(I), DEFAULT_ALIGNMENT);

    if (!mat->tile_ptr || !mat->tile_desc || !mat->tile_row_offset || !mat->csr_offset)
        return ALPHA_SPARSE_STATUS_ALLOC_FAILED;

    // Step 5: 初始化 tile_ptr
    for (I i = 0; i <= num_tiles; ++i) {
        mat->tile_ptr[i] = std::min(i * SIGMA, m);
    }

    // Step 6: 构造 tile metadata
    for (I t = 0; t < num_tiles; ++t) {
        I row_start = mat->tile_ptr[t];
        I row_end   = std::min(mat->tile_ptr[t + 1], m);

        I start_ptr = row_ptr[row_start];
        I end_ptr   = row_ptr[row_end];

        // 当前 tile 的 nnz 总数
        I nnz_in_tile = end_ptr - start_ptr;
        mat->csr_offset[t] = nnz_in_tile;

        // 描述字段初始化为 0
        I desc = 0;

        // Bit 2: 空 tile
        if (nnz_in_tile == 0)
            desc |= (1 << 2);

        // Bit 1: 多行 tile
        if ((row_end - row_start) > 1)
            desc |= (1 << 1);

        // 判断是否有跨 tile 的 partial row
        bool has_partial_row = false;
        if (t < num_tiles - 1) {
            I this_tile_last_row = mat->tile_ptr[t + 1] - 1;
            I next_tile_first_row = mat->tile_ptr[t + 1];
            if (this_tile_last_row == next_tile_first_row) {
                has_partial_row = true;
            }
        }

        // Bit 0 和 Bit 3：partial row 与 row_offset
        if (has_partial_row) {
            desc |= (1 << 0); // partial row
            desc |= (1 << 3); // 需要 row_offset
            mat->tile_row_offset[t] = row_ptr[mat->tile_ptr[t + 1]] - row_ptr[mat->tile_ptr[t]];
        } else {
            mat->tile_row_offset[t] = 0;
        }

        mat->tile_desc[t] = desc;
    }

    // Step 7: 清理中间结构（只保留 mat）
    alpha_free(csr_mat);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
