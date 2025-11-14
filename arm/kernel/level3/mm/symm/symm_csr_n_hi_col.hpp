#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/bisearch.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>

#define CACHELINE 64

template <typename J>
static void
symm_csr_n_hi_col_pack(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy, const ALPHA_INT lcs, const ALPHA_INT lce)
{
    ALPHA_INT lcl = lce - lcs;
    ALPHA_INT ldpx = lcl;
    ALPHA_INT ldpy = lcl;
    J *Xpack = (J*)alpha_malloc(mat->cols * lcl * sizeof(J));
    J *Ypack = (J*)alpha_malloc(mat->rows * lcl * sizeof(J));
    pack_matrix_col2row(mat->cols, lcl, &x[index2(lcs, 0, ldx)], ldx, Xpack, ldpx);
    pack_matrix_col2row(mat->rows, lcl, &y[index2(lcs, 0, ldy)], ldy, Ypack, ldpy);
    for (ALPHA_INT ar = 0; ar < mat->rows; ++ar)
    {
        ALPHA_INT rs = mat->row_data[ar];
        ALPHA_INT re = mat->row_data[ar+1];
        ALPHA_INT start = alpha_lower_bound(&mat->col_data[rs], &mat->col_data[re], ar) - mat->col_data;
        ALPHA_INT end = re;
        for (ALPHA_INT ai = start; ai < end; ++ai)
        {
            ALPHA_INT ac = mat->col_data[ai];
            if (ac > ar)
            {
                J val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                vec_fma2<J>(&Ypack[index2(ar, 0, ldpy)], &Xpack[index2(ac, 0, ldpx)], val, lcl);
                vec_fma2<J>(&Ypack[index2(ac, 0, ldpy)], &Xpack[index2(ar, 0, ldpx)], val, lcl);
            }
            else if (ac == ar)
            {
                J val = alpha_mul(alpha, ((J*)mat->val_data)[ai]);
                vec_fma2<J>(&Ypack[index2(ar, 0, ldpy)], &Xpack[index2(ac, 0, ldpx)], val, lcl);
            }
        }
    }
    pack_matrix_row2col(mat->rows, lcl, Ypack, ldpy, &y[index2(lcs, 0, ldy)], ldy);
    alpha_free(Xpack);
    alpha_free(Ypack);
}

template <typename J>
alphasparseStatus_t
symm_csr_n_hi_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
        ALPHA_INT num_threads = alpha_get_thread_num();
    #ifdef _OPENMP
    #pragma omp parallel for num_threads(num_threads)
    #endif
        for (ALPHA_INT c = 0; c < columns; c++)
            vec_mul2(&y[index2(c, 0, ldy)], &y[index2(c, 0, ldy)], beta, mat->rows);

        ALPHA_INT block_size = CACHELINE / sizeof(J);
        ALPHA_INT block_num = (columns + block_size - 1) / block_size;
        if (num_threads > block_num)
            num_threads = block_num;

    #ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
        {
            ALPHA_INT tid = alpha_get_thread_id();
            ALPHA_INT bcl = cross_block_low(tid, num_threads, block_num) * block_size;
            ALPHA_INT bch = cross_block_high(tid, num_threads, block_num) * block_size;
            if (bch > columns)
                bch = columns;
            symm_csr_n_hi_col_pack(alpha, mat, x, columns, ldx, beta, y, ldy, bcl, bch);
        }
        return ALPHA_SPARSE_STATUS_SUCCESS;
}
