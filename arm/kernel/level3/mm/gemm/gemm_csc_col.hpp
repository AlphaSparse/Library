#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
// #include "alphasparse/opt.h"

// #ifdef _OPENMP
// #include <omp.h>
// #endif

// #define TILE 8
// #define ALIGN(a) (((a)/TILE)*TILE)

// template <typename TYPE>
// alphasparseStatus_t gemm_csc_col(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
// {
//     ALPHA_INT num_threads = alpha_get_thread_num();
//     ALPHA_INT align_rows = ALIGN(mat->rows);
// #ifdef _OPENMP
// #pragma omp parallel for num_threads(num_threads)
// #endif
//     for (ALPHA_INT cc = 0; cc < columns; ++cc)
//     {
//         const TYPE *X = &x[index2(cc, 0, ldx)];
//         TYPE *Y = &y[index2(cc, 0, ldy)];
//         {
//             /*ALPHA_INT r;
//             ALPHA_Couble y0,y1,y2,y3;
//             for(r=0; r<align_rows; r+=TILE)
//             {
//                 y0 = Y[r]   * beta;
//                 y1 = Y[r+1] * beta;
//                 y2 = Y[r+2] * beta;
//                 y3 = Y[r+3] * beta;
//                 Y[r] = y0;
//                 Y[r+1] = y1;
//                 Y[r+2] = y2;
//                 Y[r+3] = y3;
//             }
//             for (; r < mat->rows; r++)*/
//             for(ALPHA_INT r = 0; r < mat->rows; ++r)
//                 alpha_mul(Y[r], beta, Y[r]);
//         }

//         #define xval X[br]
//         for(ALPHA_INT br = 0; br < mat->cols; ++br)
//         {
//             ALPHA_INT ai;
//             ALPHA_INT ar0, ar1, ar2, ar3, ar4, ar5, ar6, ar7;
//             ALPHA_INT align_ce = mat->cols_start[br] + ALIGN(mat->cols_end[br] - mat->cols_start[br]);
//             //TYPE xval = X[br];
//             TYPE spval0, spval1, spval2, spval3, spval4, spval5, spval6, spval7;
//             for (ai = mat->cols_start[br]; ai < align_ce; ai+=TILE)
//             {
//                 ar0 = mat->row_data[ai];
//                 ar1 = mat->row_data[ai+1];
//                 ar2 = mat->row_data[ai+2];
//                 ar3 = mat->row_data[ai+3];
//                 ar4 = mat->row_data[ai+4];
//                 ar5 = mat->row_data[ai+5];
//                 ar6 = mat->row_data[ai+6];
//                 ar7 = mat->row_data[ai+7];

//                 alpha_mul(spval0, alpha, ((TYPE *)mat->values)[ai]);
//                 alpha_mul(spval1, alpha, ((TYPE *)mat->values)[ai+1]);
//                 alpha_mul(spval2, alpha, ((TYPE *)mat->values)[ai+2]);
//                 alpha_mul(spval3, alpha, ((TYPE *)mat->values)[ai+3]);
//                 alpha_mul(spval4, alpha, ((TYPE *)mat->values)[ai+4]);
//                 alpha_mul(spval5, alpha, ((TYPE *)mat->values)[ai+5]);
//                 alpha_mul(spval6, alpha, ((TYPE *)mat->values)[ai+6]);
//                 alpha_mul(spval7, alpha, ((TYPE *)mat->values)[ai+7]);

//                 alpha_madde(Y[ar0], spval0, xval);
//                 alpha_madde(Y[ar1], spval1, xval);
//                 alpha_madde(Y[ar2], spval2, xval);
//                 alpha_madde(Y[ar3], spval3, xval);
//                 alpha_madde(Y[ar4], spval4, xval);
//                 alpha_madde(Y[ar5], spval5, xval);
//                 alpha_madde(Y[ar6], spval6, xval);
//                 alpha_madde(Y[ar7], spval7, xval);
//             }
//             for(; ai<mat->cols_end[br]; ++ai)          
//             //for (ALPHA_INT ai = mat->cols_start[br]; ai < mat->cols_end[br]; ++ai)
//             {
//                 ALPHA_INT ar = mat->row_data[ai];
//                 TYPE spval;// = alpha * ((TYPE *)mat->values)[ai];
//                 alpha_mul(spval, alpha, ((TYPE *)mat->values)[ai]);
//                 alpha_madde(Y[ar], spval, xval);
//                 //Y[ar] += spval * xval; 
//             }
//         }
//         #undef xval
//     }
//     return ALPHA_SPARSE_STATUS_SUCCESS;
// }

template <typename TYPE>
alphasparseStatus_t
gemm_csc_col(const TYPE alpha, const internal_spmat mat, const TYPE *x, const ALPHA_INT columns, const ALPHA_INT ldx, const TYPE beta, TYPE *y, const ALPHA_INT ldy)
{
    internal_spmat trans;
    trans->val_data = (mat->val_data);
    trans->row_data = mat->col_data;
    // trans->rows_end = mat->cols_end;
    trans->col_data = mat->row_data;
    trans->rows = mat->cols;
    trans->cols = mat->rows;
    gemm_csr_col_trans(alpha, trans, x, columns, ldx, beta, y, ldy);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
