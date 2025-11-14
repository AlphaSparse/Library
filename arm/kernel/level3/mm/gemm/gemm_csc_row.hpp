#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
// #include "alphasparse/opt.h"

// #include <stdbool.h>
// #include <string.h>
// #ifdef _OPENMP
// #include <omp.h>
// #endif
// template <typename TYPE>
// alphasparseStatus_t gemm_csc_row(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
// {
//     ALPHA_INT ncpu, mcpu, nblk, mblk;
//     ALPHA_INT num_threads = alpha_get_thread_num();
//     balanced_divisors2(mat->rows, columns, num_threads, &mcpu, &ncpu);
//     mblk = (mat->rows + mcpu-1) / mcpu;
//     nblk = (columns + ncpu-1) / ncpu;
// #ifdef _OPENMP
// #pragma omp parallel num_threads(num_threads)
// #endif
//     {
//         ALPHA_INT tid   = alpha_get_thread_id();
//         ALPHA_INT tid_x = tid % mcpu;
//         ALPHA_INT tid_y = tid / mcpu;
//         ALPHA_INT lrs   = tid_x * mblk;
//         ALPHA_INT lre   = alpha_min((tid_x+1)*mblk, mat->rows);
//         ALPHA_INT lcs   = tid_y * nblk;
//         ALPHA_INT lce   = alpha_min((tid_y+1)*nblk, columns);

//         for(ALPHA_INT ix = lrs; ix < lre; ++ix)
//         for(ALPHA_INT iy = lcs; iy < lce; ++iy)
//             alpha_mul(y[index2(ix,iy,ldy)], beta, y[index2(ix,iy,ldy)]);
//             // = beta;

//         for(ALPHA_INT ac = 0; ac < mat->cols; ++ac)
//         {
//             for(ALPHA_INT ai=mat->cols_start[ac]; ai < mat->cols_end[ac]; ++ai)
//             {
//                 if(mat->row_data[ai] < lrs) continue;
//                 if(mat->row_data[ai] >= lre) break;

//                 ALPHA_INT          ar  = mat->row_data[ai];
//                 TYPE       *Y  = &y[index2(ar, 0, ldy)];
//                 const TYPE *X  = &x[index2(ac, 0, ldx)];

//                 TYPE       val;// = alpha * ((TYPE *)mat->values)[ai];
//                 alpha_mul(val, alpha, ((TYPE *)mat->values)[ai]);

//                 for(ALPHA_INT cc = lcs; cc < lce; ++cc)
//                     alpha_madde(Y[cc], val, X[cc]);
//                     //Y[cc] += val * X[cc];
//             }
//         }
//     }
//     return ALPHA_SPARSE_STATUS_SUCCESS;
// }

template <typename TYPE>
alphasparseStatus_t
gemm_csc_row(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat trans;
    trans->val_data = mat->val_data;
    trans->row_data = mat->col_data;
    // trans->rows_end = mat->cols_end;
    trans->col_data = mat->row_data;
    trans->rows = mat->cols;
    trans->cols = mat->rows;
    gemm_csr_row_trans(alpha, trans, x, columns, ldx, beta, y, ldy);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
