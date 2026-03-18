#include <immintrin.h>
#include <stddef.h>
#include <immintrin.h>

/**
//  * @brief SELL-C-σ 矩阵向量乘：double 版，用 AVX 做 4 路并行
//  *
//  * y[orig_row] = alpha * (A_row · x) + beta * y[orig_row]
//  *
//  * @param alpha          缩放 A·x 的系数
//  * @param beta           缩放原 y 的系数
//  * @param slice_ptr      每个块（每 C 行）在 values/col_idx 中的起始偏移
//  * @param col_idx        SELL 格式下的列索引数组（已填 0 作为 padding）
//  * @param values         SELL 格式下的矩阵值数组（按块展开，每块是 C 列对齐）
//  * @param block_max_nnz  每个块的行最大非零长度（即你叫的 block_max_nnz）
//  * @param x              输入向量 x
//  * @param y              输入/输出向量 y
//  * @param C              SELL 中的 C（每块行数）
//  * @param sigma          σ（未在本算子中用到，但保留签名一致）
//  * @param slice_start    从哪个块号开始处理（包含）
//  * @param slice_end      到哪个块号结束（不含）
//  * @param row_idx_map    排序后行号恢复到原始行号的映射数组
 */
void __spmv_sell_csigma_serial_host_avx_double(
    const double  alpha,
    const double  beta,
    const int    *slice_ptr,
    const int    *col_idx,
    const double *values,
    const int    *block_max_nnz,
    const double *x,
    double       *y,
    const int     C,
    const int     sigma,
    const int     slice_start,
    const int     slice_end,
    const int    *row_idx_map)
{
    const int group = 4;  // AVX 一次处理 4 个 double

    // 遍历每个块
    for (int br = slice_start; br < slice_end; ++br) {
        int offset = slice_ptr[br];
        int ml     = block_max_nnz[br];      // 该块中行的实际长度

        // 每块有 C 列，分成 C/group 组
        for (int j = 0; j < C; j += group) {
            __m256d acc = _mm256_setzero_pd();

            // 对该组内的每一行做乘加
            for (int i = 0; i < ml; ++i) {
                int base = offset + i * C + j;
                // load 4 个矩阵元素
                __m256d mat = _mm256_loadu_pd(values + base);
                // load 4 个 x 元素（padding 时 col_idx 已填 0）
                __m256d vec = _mm256_set_pd(
                    x[col_idx[base + 3]],
                    x[col_idx[base + 2]],
                    x[col_idx[base + 1]],
                    x[col_idx[base + 0]]
                );
                // FMA
                acc = _mm256_fmadd_pd(mat, vec, acc);
            }

            // 把累加结果写入临时数组
            double tmp[4];
            _mm256_storeu_pd(tmp, acc);

            // 对应的 4 行（排序后行号）累加到 y
            int base_row = br * C + j;
            for (int t = 0; t < group; ++t) {
                int row       = base_row + t;
                int orig_row  = row_idx_map[row];
                // y = alpha * sum + beta * y
                y[orig_row] = alpha * tmp[t] + beta * y[orig_row];
            }
        }
    }
}

/**
 * @brief SELL-C-σ 矩阵向量乘：float 版，用 AVX 做 8 路并行
 */
void __spmv_sell_csigma_serial_host_avx_float(
    const float   alpha,
    const float   beta,
    const int    *slice_ptr,
    const int    *col_idx,
    const float  *values,
    const int    *block_max_nnz,
    const float  *x,
    float        *y,
    const int     C,
    const int     sigma,
    const int     slice_start,
    const int     slice_end,
    const int    *row_idx_map)
{
    const int group = 8;  // AVX 一次处理 8 个 float

    for (int br = slice_start; br < slice_end; ++br) {
        int offset = slice_ptr[br];
        int ml     = block_max_nnz[br];

        for (int j = 0; j < C; j += group) {
            __m256 acc = _mm256_setzero_ps();

            for (int i = 0; i < ml; ++i) {
                int base = offset + i * C + j;
                __m256 mat = _mm256_loadu_ps(values + base);
                __m256 vec = _mm256_set_ps(
                    x[col_idx[base + 7]],
                    x[col_idx[base + 6]],
                    x[col_idx[base + 5]],
                    x[col_idx[base + 4]],
                    x[col_idx[base + 3]],
                    x[col_idx[base + 2]],
                    x[col_idx[base + 1]],
                    x[col_idx[base + 0]]
                );
                acc = _mm256_fmadd_ps(mat, vec, acc);
            }

            float tmp[8];
            _mm256_storeu_ps(tmp, acc);

            int base_row = br * C + j;
            for (int t = 0; t < group; ++t) {
                int row      = base_row + t;
                int orig_row = row_idx_map[row];
                y[orig_row] = alpha * tmp[t] + beta * y[orig_row];
            }
        }
    }
}

// // double 版
// void __spmv_sell_csigma_serial_host_avx_double(
//     const double    alpha,
//     const double    beta,
//     const int      *slice_ptr,     
//     const int      *col_idx,       
//     const double   *values,       
//     const int      *block_max_nnz,       
//     const double   *x,             
//     double         *y,             
//     const int       C,             
//     const int       sigma,
//     const int       slice_start,   
//     const int       slice_end,     
//     const int      *row_idx_map    
// ) {
//     int groups = C / 4;  // 每组用一个 256‑bit 寄存器处理 4 个 double
//     for (int br = slice_start; br < slice_end; ++br) {
//         int base_ptr = slice_ptr[br];      // 本 block 在 values/col_idx 的起始偏移（按行）
//         // 注意 row_idx_map 将 [br*C + i] 映射回原始行号
//         for (int i = 0; i < slice_height; ++i) {
//             int reordered_row = br * C + i;
//             int orig_row       = row_idx_map[reordered_row];
//             __m256d acc_vec    = _mm256_setzero_pd();

//             // 对每个 group（4 列一组）做向量化乘加
//             for (int g = 0; g < groups; ++g) {
//                 int off = base_ptr + i * C + g * 4;
//                 __m256d m = _mm256_loadu_pd(values + off);
//                 // 直接加载 x[col_idx[...] ]，padding 时值 = 0
//                 __m256d v = _mm256_set_pd(
//                     x[col_idx[off + 3]],
//                     x[col_idx[off + 2]],
//                     x[col_idx[off + 1]],
//                     x[col_idx[off + 0]]
//                 );
//                 acc_vec = _mm256_fmadd_pd(m, v, acc_vec);
//             }

//             // 将 4 路累加向量展开求和
//             double tmp[4];
//             _mm256_storeu_pd(tmp, acc_vec);
//             double sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

//             // alpha * sum + beta * y[orig_row]
//             y[orig_row] = alpha * sum + beta * y[orig_row];
//         }
//     }
// }

// // float 版
// void __spmv_sell_csigma_serial_host_avx_float(
//     const float     alpha,
//     const float     beta,
//     const int      *slice_ptr,
//     const int      *col_idx,
//     const float    *values,
//     const float    *x,
//     float          *y,
//     const int       C,           
//     const int       sigma,
//     const int       slice_start,
//     const int       slice_end,
//     const int      *row_idx_map
// ) {
//     int groups = C / 8;  // 每组 8 个 float
//     for (int br = slice_start; br < slice_end; ++br) {
//         int base_ptr = slice_ptr[br];
//         for (int i = 0; i < slice_height; ++i) {
//             int reordered_row = br * C + i;
//             int orig_row       = row_idx_map[reordered_row];
//             __m256 acc_vec     = _mm256_setzero_ps();

//             for (int g = 0; g < groups; ++g) {
//                 int off = base_ptr + i * C + g * 8;
//                 __m256 m = _mm256_loadu_ps(values + off);
//                 __m256 v = _mm256_set_ps(
//                     x[col_idx[off + 7]],
//                     x[col_idx[off + 6]],
//                     x[col_idx[off + 5]],
//                     x[col_idx[off + 4]],
//                     x[col_idx[off + 3]],
//                     x[col_idx[off + 2]],
//                     x[col_idx[off + 1]],
//                     x[col_idx[off + 0]]
//                 );
//                 acc_vec = _mm256_fmadd_ps(m, v, acc_vec);
//             }

//             float tmp[8];
//             _mm256_storeu_ps(tmp, acc_vec);
//             float sum = 0;
//             for (int j = 0; j < 8; ++j) sum += tmp[j];

//             y[orig_row] = alpha * sum + beta * y[orig_row];
//         }
//     }
// }
