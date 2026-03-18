#include <alphasparse/opt.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "alphasparse/format.h"
#include "alphasparse/util.h"
#include "alphasparse/util/prefix_sum.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/util/partition.h"
#include <alphasparse/opt.h>



template <typename I, typename J, typename T>
alphasparseStatus_t convert_sell_csigma_csr(const T *source,
                                            const ALPHA_INT C,
                                            const ALPHA_INT SIGMA,
                                            T **dest)
{
    typedef struct {
        I idx;
        I len;
    } RowLen;

    I rows = source->rows;
    I cols = source->cols;
    I nnz  = source->nnz;

    I *csr_row_ptr = source->row_data;
    I *csr_col_idx = source->col_data;
    J *csr_values  = (J*)source->val_data;

    // 创建目标矩阵
    T *mat = (T*)alpha_malloc(sizeof(T));
    if (!mat) return ALPHA_SPARSE_STATUS_ALLOC_FAILED;
    std::memset(mat, 0, sizeof(T));
    *dest = mat;

    mat->rows = rows;
    mat->cols = cols;
    mat->nnz  = nnz;
    mat->sell_C = C;
    mat->sell_sigma = SIGMA;

    const I n_block_rows = (rows + C - 1) / C;
    mat->sell_blocks = n_block_rows;

    // ========== 1. 每行长度 ==========
    std::vector<I> row_len(rows);
    for (I i = 0; i < rows; ++i)
        row_len[i] = csr_row_ptr[i+1] - csr_row_ptr[i];

    // ========== 2. 生成 reorder 行顺序 ==========
    mat->reorders = (I*)alpha_malloc(sizeof(I) * rows);
    if (!mat->reorders) return ALPHA_SPARSE_STATUS_ALLOC_FAILED;
    for (I i = 0; i < rows; ++i) mat->reorders[i] = i;

    if (SIGMA > 1) {
        RowLen *group = (RowLen*)alpha_malloc(sizeof(RowLen) * SIGMA);
        for (I s = 0; s < rows; s += SIGMA) {
            I e = std::min(s + SIGMA, rows);
            I len = e - s;
            for (I i = 0; i < len; ++i) {
                I row = mat->reorders[s + i];
                group[i].idx = row;
                group[i].len = row_len[row];
            }
            std::sort(group, group + len, [](const RowLen &a, const RowLen &b) {
                return a.len > b.len;
            });
            for (I i = 0; i < len; ++i)
                mat->reorders[s + i] = group[i].idx;
        }
        alpha_free(group);
    } else if (SIGMA <= 0) {
        std::vector<RowLen> all(rows);
        for (I i = 0; i < rows; ++i) {
            all[i].idx = mat->reorders[i];
            all[i].len = row_len[all[i].idx];
        }
        std::sort(all.begin(), all.end(), [](const RowLen &a, const RowLen &b) {
            return a.len > b.len;
        });
        for (I i = 0; i < rows; ++i)
            mat->reorders[i] = all[i].idx;
    }

    // ========== 3. 分块最大行长 & row_ptr ==========
    mat->block_max_nnz = (I*)alpha_malloc(sizeof(I) * n_block_rows);
    mat->pointers = (I*)alpha_malloc(sizeof(I) * (n_block_rows + 1));
    if (!mat->block_max_nnz || !mat->pointers)
        return ALPHA_SPARSE_STATUS_ALLOC_FAILED;

    mat->pointers[0] = 0;
    I nnz_blocks = 0;
    for (I br = 0; br < n_block_rows; ++br) {
        I maxl = 0;
        for (I j = 0; j < C; ++j) {
            I idx = br * C + j;
            if (idx >= rows) break;
            I row = mat->reorders[idx];
            if (row_len[row] > maxl) maxl = row_len[row];
        }
        mat->block_max_nnz[br] = maxl;
        mat->pointers[br + 1] = mat->pointers[br] + maxl * C;
        nnz_blocks += maxl * C;
    }
    mat->sell_nnz = nnz_blocks;

    // ========== 4. 分配 col_data 和 val_data ==========
    mat->col_data = (I*)alpha_malloc(sizeof(I) * nnz_blocks);
    mat->val_data = (J*)alpha_malloc(sizeof(J) * nnz_blocks);
    if (!mat->col_data || !mat->val_data)
        return ALPHA_SPARSE_STATUS_ALLOC_FAILED;

    std::fill(mat->col_data, mat->col_data + nnz_blocks, -1);
    std::fill((J*)mat->val_data, (J*)mat->val_data + nnz_blocks, (J)0);

    // ========== 5. 填充 SELL 数据 ==========
    for (I br = 0; br < n_block_rows; ++br) {
        I maxl = mat->block_max_nnz[br];
        I base = mat->pointers[br];
        for (I i = 0; i < maxl; ++i) {
            for (I j = 0; j < C; ++j) {
                I global_row = br * C + j;
                if (global_row >= rows) continue;

                I orig_row = mat->reorders[global_row];
                I start = csr_row_ptr[orig_row];
                I len   = csr_row_ptr[orig_row + 1] - start;
                I offset = base + i * C + j;

                if (i < len) {
                    mat->col_data[offset] = csr_col_idx[start + i];
                    ((J*)mat->val_data)[offset] = csr_values[start + i];
                }
            }
        }
    }

    // // ========== 6. 打印 SELL 格式内容 ==========
    // printf("SELL-C-sigma Format:\n");
    // printf("rows = %ld, cols = %ld, nnz = %ld\n", mat->rows, mat->cols, mat->nnz);
    // printf("C = %ld, sigma = %ld, blocks = %ld, sell_nnz = %ld\n", mat->sell_C, mat->sell_sigma, mat->sell_blocks, mat->sell_nnz);

    // printf("row_ptr (pointers):\n");
    // for (I i = 0; i <= n_block_rows; ++i)
    //     printf("%d ", mat->pointers[i]);
    // printf("\n");

    // printf("block_max_nnz:\n");
    // for (I i = 0; i < n_block_rows; ++i)
    //     printf("%d ", mat->block_max_nnz[i]);
    // printf("\n");

    // printf("reorders:\n");
    // for (I i = 0; i < rows; ++i)
    //     printf("%d ", mat->reorders[i]);
    // printf("\n");

    // printf("col_data:\n");
    // for (I i = 0; i < nnz_blocks; ++i)
    //     printf("%d ", mat->col_data[i]);
    // printf("\n");

    // printf("val_data:\n");
    // for (I i = 0; i < nnz_blocks; ++i)
    //     printf("%.2f ", ((J*)mat->val_data)[i]);
    // printf("\n");

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
