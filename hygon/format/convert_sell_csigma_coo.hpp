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


template <typename I, typename J, typename T>
alphasparseStatus_t convert_sell_csigma_coo(const T *source,
                                            const ALPHA_INT C,
                                            const ALPHA_INT SIGMA,
                                            T **dest) {
    // printf("Begin coo to sell\n");
    T *mat = (T*)alpha_malloc(sizeof(T));
    if (!mat) return ALPHA_SPARSE_STATUS_ALLOC_FAILED;
    *dest = mat;

    I m = source->rows;
    I n = source->cols;
    I nnz = source->nnz;
    mat->rows = m;
    mat->cols = n;
    mat->sell_C = C;
    mat->sell_sigma = SIGMA;

    // Step 1: Count row nnz
    std::vector<I> row_nnz(m, 0); 
    for (I i = 0; i < nnz; ++i)
        row_nnz[source->row_data[i]]++;

    // Step 2: Sort row_ids within blocks
    std::vector<I> row_ids(m);
    for (I i = 0; i < m; ++i) row_ids[i] = i;
    if (SIGMA > 1) {
        for (I i = 0; i < m; i += SIGMA) {
            I end = std::min(i + SIGMA, m);
            std::sort(row_ids.begin() + i, row_ids.begin() + end, [&](I a, I b) {
                return row_nnz[a] > row_nnz[b];
            });
        }
    } else if (SIGMA <= 0) {
        std::sort(row_ids.begin(), row_ids.end(), [&](I a, I b) {
            return row_nnz[a] > row_nnz[b];
        });
    }

    // Step 3: Preprocess COO (sorted by row)
    std::vector<I> perm(nnz);
    for (I i = 0; i < nnz; ++i) perm[i] = i;
    std::sort(perm.begin(), perm.end(), [&](I a, I b) {
        if (source->row_data[a] != source->row_data[b])
            return source->row_data[a] < source->row_data[b];
        return source->col_data[a] < source->col_data[b];
    });
    std::vector<I> sorted_row(nnz), sorted_col(nnz);
    std::vector<J> sorted_val(nnz);
    for (I i = 0; i < nnz; ++i) {
        sorted_row[i] = source->row_data[perm[i]];
        sorted_col[i] = source->col_data[perm[i]];
        sorted_val[i] = ((J*)source->val_data)[perm[i]];
    }

    // Step 4: Build coo_row_ptr
    std::vector<I> coo_row_ptr(m + 1, 0);
    for (I i = 0; i < nnz; ++i) coo_row_ptr[sorted_row[i] + 1]++;
    for (I i = 1; i <= m; ++i) coo_row_ptr[i] += coo_row_ptr[i - 1];

    // Step 5: Determine block layout
    I n_block_rows = (m + C - 1) / C;
    std::vector<I> max_row_nnz(n_block_rows, 0);
    for (I br = 0; br < n_block_rows; ++br) {
        I start = br * C;
        I end = std::min(start + C, m);
        for (I r = start; r < end; ++r) {
            I cnt = row_nnz[row_ids[r]];
            if (cnt > max_row_nnz[br]) max_row_nnz[br] = cnt;
        }
    }
    mat->block_max_nnz = (I*)alpha_memalign(n_block_rows * sizeof(I), DEFAULT_ALIGNMENT);
    if (!mat->block_max_nnz) return ALPHA_SPARSE_STATUS_ALLOC_FAILED;
    memcpy(mat->block_max_nnz, max_row_nnz.data(), n_block_rows * sizeof(I));

    I nnz_blocks = 0;
    for (I br = 0; br < n_block_rows; ++br) nnz_blocks += max_row_nnz[br];

    mat->sell_blocks = n_block_rows;
    mat->sell_nnz = nnz_blocks;
    mat->row_data = (I*)alpha_memalign((n_block_rows + 1) * sizeof(I), DEFAULT_ALIGNMENT);
    mat->col_data = (I*)alpha_memalign((nnz_blocks * C) * sizeof(I), DEFAULT_ALIGNMENT);
    mat->val_data = (J*)alpha_memalign((nnz_blocks * C) * sizeof(J), DEFAULT_ALIGNMENT);
    mat->reorders = (I*)alpha_memalign(m * sizeof(I), DEFAULT_ALIGNMENT);

    I *row_ptr = mat->row_data;
    I *col_idx = mat->col_data;
    J *val_ptr = (J*)mat->val_data;
    I *orders = mat->reorders;

    row_ptr[0] = 0;
    for (I i = 0; i < m; ++i) orders[i] = row_ids[i];
    for (I br = 0; br < n_block_rows; ++br) {
        row_ptr[br + 1] = row_ptr[br] + max_row_nnz[br] * C;
    }

    // printf("Begin padding...\n");
    for (I br = 0; br < n_block_rows; ++br) {
        I start = br * C;
        I end = std::min(start + C, m);
        I max_nz = max_row_nnz[br];
        I blk_off = row_ptr[br];

        for (I k = 0; k < max_nz; ++k) {
            for (I r = start; r < end; ++r) {
                I orig = row_ids[r];
                I pos = blk_off + k * C + (r - start);  // <-- FIXED HERE
                I row_start = coo_row_ptr[orig];
                I row_end = coo_row_ptr[orig + 1];
                I row_len = row_end - row_start;

                if (k < row_len) {
                    I idx = row_start + k;
                    col_idx[pos] = sorted_col[idx];
                    val_ptr[pos] = sorted_val[idx];
                } else {
                    col_idx[pos] = 0;
                    val_ptr[pos] = (J)0;
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
