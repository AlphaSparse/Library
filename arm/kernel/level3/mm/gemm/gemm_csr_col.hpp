#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/util/partition.h"
#include "alphasparse/util/pack.h"

#include "util/vec_dot.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <memory.h>

#define Ntile 12
#define Mtile 64
#define Ktile 4096

#define COL_BLOCK 4096
#define ROW_BLOCK 64

static void gemm_s_csr_ntile_ktile_mtile_unroll4(const float alpha, const internal_spmat mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy, ALPHA_INT lrs, ALPHA_INT lre)
{
    ALPHA_INT *pos;
    ALPHA_INT bcl, ldp;
    csr_col_partition(mat, lrs, lre, Ktile, &pos, &bcl, &ldp);

    const ALPHA_INT NTILE = 4;
    float tmp[NTILE];
    ALPHA_INT columns4 = columns - NTILE + 1;
    ALPHA_INT nbs = 0;
    for (; nbs < columns4; nbs += NTILE)
    {
        const float *Xtile0 = &x[index2(nbs, 0, ldx)];
        const float *Xtile1 = &x[index2(nbs + 1, 0, ldx)];
        const float *Xtile2 = &x[index2(nbs + 2, 0, ldx)];
        const float *Xtile3 = &x[index2(nbs + 3, 0, ldx)];
        float *Ytile0 = &y[index2(nbs, 0, ldy)];
        float *Ytile1 = &y[index2(nbs + 1, 0, ldy)];
        float *Ytile2 = &y[index2(nbs + 2, 0, ldy)];
        float *Ytile3 = &y[index2(nbs + 3, 0, ldy)];
        for (ALPHA_INT brs = lrs; brs < lre; brs += Mtile)
        {
            ALPHA_INT bre = alpha_min(lre, brs + Mtile);
            for (ALPHA_INT i = brs; i < bre; i++)
            {
                Ytile0[i] *= beta;
                Ytile1[i] *= beta;
                Ytile2[i] *= beta;
                Ytile3[i] *= beta;
            }
            for (ALPHA_INT bi = 0; bi < bcl; bi++)
            {
                for (ALPHA_INT r = brs; r < bre; r++)
                {
                    float tmp0 = 0.f, tmp1 = 0.f, tmp2 = 0.f, tmp3 = 0.f;
                    ALPHA_INT bis = pos[index2(r - lrs, bi, ldp)];
                    ALPHA_INT bie = pos[index2(r - lrs, bi + 1, ldp)];
                    ALPHA_INT bil = bie - bis;
                    float *A_val = &((float *)mat->val_data)[bis];
                    ALPHA_INT *A_col = &mat->col_data[bis];
                    for (ALPHA_INT bai = 0; bai < bil; bai++)
                    {
                        ALPHA_INT col = A_col[bai];
                        float val = A_val[bai];
                        tmp0 += val * Xtile0[col];
                        tmp1 += val * Xtile1[col];
                        tmp2 += val * Xtile2[col];
                        tmp3 += val * Xtile3[col];
                    }
                    Ytile0[r] += tmp0 * alpha;
                    Ytile1[r] += tmp1 * alpha;
                    Ytile2[r] += tmp2 * alpha;
                    Ytile3[r] += tmp3 * alpha;
                }
            }
        }
    }
    for (; nbs < columns; nbs += 1)
    {
        const float *Xtile0 = &x[index2(nbs, 0, ldx)];
        float *Ytile0 = &y[index2(nbs, 0, ldy)];
        for (ALPHA_INT brs = lrs; brs < lre; brs += Mtile)
        {
            ALPHA_INT bre = alpha_min(lre, brs + Mtile);
            for (ALPHA_INT i = brs; i < bre; i++)
            {
                Ytile0[i] *= beta;
            }
            for (ALPHA_INT bi = 0; bi < bcl; bi++)
            {
                for (ALPHA_INT r = brs; r < bre; r++)
                {
                    float tmp0 = 0.f;
                    ALPHA_INT bis = pos[index2(r - lrs, bi, ldp)];
                    ALPHA_INT bie = pos[index2(r - lrs, bi + 1, ldp)];
                    ALPHA_INT bil = bie - bis;
                    float *A_val = &((float *)mat->val_data)[bis];
                    ALPHA_INT *A_col = &mat->col_data[bis];
                    for (ALPHA_INT bai = 0; bai < bil; bai++)
                    {
                        ALPHA_INT col = A_col[bai];
                        float val = A_val[bai];
                        tmp0 += val * Xtile0[col];
                    }
                    Ytile0[r] += tmp0 * alpha;
                }
            }
        }
    }
    alpha_free(pos);
}

static void gemm_s_csr_col_naive(const float alpha, const internal_spmat mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy, ALPHA_INT local_m_s, ALPHA_INT local_m_e)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = local_m_s; cr < local_m_e; ++cr)
        {
            float ctmp = 0;
            for (ALPHA_INT ai = mat->row_data[cr]; ai < mat->row_data[cr+1]; ++ai)
            {
                ctmp += ((float *)mat->val_data)[ai] * x[index2(cc, mat->col_data[ai], ldx)];
            }
            y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * ctmp;
        }
    }
}

static void gemm_s_csr_col_omp(const float alpha, const internal_spmat mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->row_data+1, m, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];
        gemm_s_csr_ntile_ktile_mtile_unroll4(alpha, mat, x, columns, ldx, beta, y, ldy, local_m_s, local_m_e);
        // gemm_s_csr_col_block_c4_unroll4(alpha, mat, x, columns, ldx, beta, y, ldy, local_m_s, local_m_e);
    }
}

// alphasparseStatus_t gemm_s_csr_col(const float alpha, const spmat_csr_s_t *mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy)
// {
//     gemm_s_csr_col_omp(alpha, mat, x, columns, ldx, beta, y, ldy);
//     return ALPHA_SPARSE_STATUS_SUCCESS;
// }

typedef struct
{
    ALPHA_INT row;
    ALPHA_INT length;
} row_length_s_t;

int row_length_s_cmp(const row_length_s_t *a, const row_length_s_t *b)
{
    return a->length - b->length;
}

#define CS 4

typedef struct
{
    ALPHA_INT chuck_size;
    ALPHA_INT chuck_length;
    ALPHA_INT rows[CS];
    ALPHA_INT *__restrict cols; // chuck_size * chuck_length
    float *__restrict values;
} chunk_s_t;

static void csr_s_create_ccf(chunk_s_t **chunks_p, ALPHA_INT *chunk_num_p, const internal_spmat mat)
{
    ALPHA_INT rowA = mat->rows;
    row_length_s_t *row_length = (row_length_s_t *)alpha_malloc(sizeof(row_length_s_t) * mat->rows);
    for (ALPHA_INT r = 0; r < rowA; ++r)
    {
        row_length[r].row = r;
        row_length[r].length = mat->row_data[r+1] - mat->row_data[r];
    }
    qsort(row_length, rowA, sizeof(row_length_s_t), (__compar_fn_t)row_length_s_cmp);
    ALPHA_INT bin_num = 0;
    ALPHA_INT last_len = 0;
    if (rowA > 0)
    {
        last_len = row_length[0].length;
        bin_num += 1;
    }
    for (ALPHA_INT i = 1; i < rowA; ++i)
    {
        if (last_len != row_length[i].length)
        {
            bin_num += 1;
            last_len = row_length[i].length;
        }
    }
    ALPHA_INT *bin_offset = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (bin_num + 1));
    last_len = rowA > 0 ? row_length[0].length : 0;
    bin_offset[0] = 0;
    ALPHA_INT bin_index = 0;
    for (ALPHA_INT i = 1; i < rowA; ++i)
    {
        if (last_len != row_length[i].length)
        {
            bin_index += 1;
            last_len = row_length[i].length;
            bin_offset[bin_index] = i;
        }
    }
    bin_offset[bin_num] = rowA;
    ALPHA_INT chunk_num = 0;
    for (ALPHA_INT bin_id = 0; bin_id < bin_num; bin_id++)
    {
        ALPHA_INT bin_item_num = bin_offset[bin_id + 1] - bin_offset[bin_id];
        chunk_num += (bin_item_num + CS - 1) / CS;
    }
    chunk_s_t *chunks = (chunk_s_t *)alpha_malloc(sizeof(chunk_s_t) * chunk_num);
    ALPHA_INT chunk_index = 0;
    for (ALPHA_INT bin_id = 0; bin_id < bin_num; bin_id++)
    {
        ALPHA_INT bin_start = bin_offset[bin_id];
        ALPHA_INT bin_end = bin_offset[bin_id + 1];
        for (ALPHA_INT chunk_start = bin_start; chunk_start < bin_end; chunk_start += CS)
        {
            ALPHA_INT chunk_end = alpha_min(chunk_start + CS, bin_end);
            chunk_s_t *chunk = &chunks[chunk_index];
            chunk->chuck_size = chunk_end - chunk_start;
            chunk->chuck_length = row_length[bin_start].length;
            for (ALPHA_INT row_index = chunk_start; row_index < chunk_end; row_index++)
            {
                chunk->rows[row_index - chunk_start] = row_length[row_index].row;
            }
            chunk_index += 1;
        }
    }

    ALPHA_INT eles[chunk_num];
    if (chunk_num > 0)
    {
        eles[0] = chunks[0].chuck_length * chunks[0].chuck_size;
    }
    for (ALPHA_INT i = 1; i < chunk_num; i++)
    {
        eles[i] = eles[i - 1] + chunks[i].chuck_length * chunks[i].chuck_size;
    }

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(eles, chunk_num, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT chunk_s = partition[tid];
        ALPHA_INT chunk_e = partition[tid + 1];
        for (ALPHA_INT chunk_id = chunk_s; chunk_id < chunk_e; chunk_id++)
        {
            chunk_s_t *chunk = &chunks[chunk_id];
            chunk->cols = (ALPHA_INT *)alpha_malloc(chunk->chuck_size * chunk->chuck_length * sizeof(ALPHA_INT));
            chunk->values = (float *)alpha_malloc(chunk->chuck_size * chunk->chuck_length * sizeof(float));
            for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
            {
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    ALPHA_INT row = chunk->rows[r];
                    ALPHA_INT csr_index = mat->row_data[row] + l;
                    chunk->cols[index2(l, r, chunk->chuck_size)] = mat->col_data[csr_index];
                    chunk->values[index2(l, r, chunk->chuck_size)] = ((float *)mat->val_data)[csr_index];
                }
            }
        }
    }
    alpha_free(row_length);
    alpha_free(bin_offset);
    *chunks_p = chunks;
    *chunk_num_p = chunk_num;
}

static alphasparseStatus_t
gemm_s_csr_col_chunk(const float alpha, const internal_spmat mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy)
{
    ALPHA_INT chunk_num;
    chunk_s_t *chunks;
    csr_s_create_ccf(&chunks, &chunk_num, mat);
    // compute start
    for (ALPHA_INT col = 0; col < columns; col++)
    {
        const float *X = &x[index2(col, 0, ldx)];
        float *Y = &y[index2(col, 0, ldy)];
        for (ALPHA_INT chunk_id = 0; chunk_id < chunk_num; ++chunk_id)
        {
            chunk_s_t *__restrict chunk = &chunks[chunk_id];
            float tmpY[CS];
            float tmpX[CS] = {0.f};
            ALPHA_INT tmpC[CS] = {0};
            float tmpV[CS] = {0.f};
            memset(tmpY, '\0', CS * sizeof(float));
            for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
            {
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    tmpC[r] = chunk->cols[index2(l, r, chunk->chuck_size)];
                    tmpV[r] = chunk->values[index2(l, r, chunk->chuck_size)];
                }
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    tmpX[r] = X[tmpC[r]];
                }
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    tmpY[r] += tmpX[r] * tmpV[r];
                }
            }
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                ALPHA_INT row = chunk->rows[r];
                Y[row] = Y[row] * beta + tmpY[r] * alpha;
            }
        }
    }
    for (ALPHA_INT i = 0; i < chunk_num; i++)
    {
        alpha_free(chunks[i].cols);
        alpha_free(chunks[i].values);
    }
    alpha_free(chunks);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

void gemm_s_csr_col_chunk_unroll_col4_kernel(const chunk_s_t *chunk, const float *X0, const float *X1, const float *X2, const float *X3,
                                             float *__restrict Y0, float *__restrict Y1, float *__restrict Y2, float *__restrict Y3,
                                             const float alpha, const float beta)
{
    float tmpY[4][CS];
    float tmpX[4][CS] = {{0.f}};
    ALPHA_INT tmpC[CS] = {0};
    float tmpV[CS] = {0.f};
    memset(tmpY, '\0', 4 * CS * sizeof(float));
    if (chunk->chuck_size == CS)
    {
        for (ALPHA_INT l = 0, index = 0; l < chunk->chuck_length; ++l, index += CS)
        {
            tmpC[0] = chunk->cols[index];
            tmpC[1] = chunk->cols[index + 1];
            tmpC[2] = chunk->cols[index + 2];
            tmpC[3] = chunk->cols[index + 3];
            tmpV[0] = chunk->values[index];
            tmpV[1] = chunk->values[index + 1];
            tmpV[2] = chunk->values[index + 2];
            tmpV[3] = chunk->values[index + 3];
            tmpX[0][0] = X0[tmpC[0]];
            tmpX[0][1] = X0[tmpC[1]];
            tmpX[0][2] = X0[tmpC[2]];
            tmpX[0][3] = X0[tmpC[3]];
            tmpX[1][0] = X1[tmpC[0]];
            tmpX[1][1] = X1[tmpC[1]];
            tmpX[1][2] = X1[tmpC[2]];
            tmpX[1][3] = X1[tmpC[3]];
            tmpX[2][0] = X2[tmpC[0]];
            tmpX[2][1] = X2[tmpC[1]];
            tmpX[2][2] = X2[tmpC[2]];
            tmpX[2][3] = X2[tmpC[3]];
            tmpX[3][0] = X3[tmpC[0]];
            tmpX[3][1] = X3[tmpC[1]];
            tmpX[3][2] = X3[tmpC[2]];
            tmpX[3][3] = X3[tmpC[3]];
            tmpY[0][0] += tmpX[0][0] * tmpV[0];
            tmpY[0][1] += tmpX[0][1] * tmpV[1];
            tmpY[0][2] += tmpX[0][2] * tmpV[2];
            tmpY[0][3] += tmpX[0][3] * tmpV[3];
            tmpY[1][0] += tmpX[1][0] * tmpV[0];
            tmpY[1][1] += tmpX[1][1] * tmpV[1];
            tmpY[1][2] += tmpX[1][2] * tmpV[2];
            tmpY[1][3] += tmpX[1][3] * tmpV[3];
            tmpY[2][0] += tmpX[2][0] * tmpV[0];
            tmpY[2][1] += tmpX[2][1] * tmpV[1];
            tmpY[2][2] += tmpX[2][2] * tmpV[2];
            tmpY[2][3] += tmpX[2][3] * tmpV[3];
            tmpY[3][0] += tmpX[3][0] * tmpV[0];
            tmpY[3][1] += tmpX[3][1] * tmpV[1];
            tmpY[3][2] += tmpX[3][2] * tmpV[2];
            tmpY[3][3] += tmpX[3][3] * tmpV[3];
        }
    }
    else
    {
        for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
        {
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                tmpC[r] = chunk->cols[index2(l, r, chunk->chuck_size)];
                tmpV[r] = chunk->values[index2(l, r, chunk->chuck_size)];
            }
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                tmpX[0][r] = X0[tmpC[r]];
                tmpX[1][r] = X1[tmpC[r]];
                tmpX[2][r] = X2[tmpC[r]];
                tmpX[3][r] = X3[tmpC[r]];
            }
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                tmpY[0][r] += tmpX[0][r] * tmpV[r];
                tmpY[1][r] += tmpX[1][r] * tmpV[r];
                tmpY[2][r] += tmpX[2][r] * tmpV[r];
                tmpY[3][r] += tmpX[3][r] * tmpV[r];
            }
        }
    }
    for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
    {
        ALPHA_INT row = chunk->rows[r];
        Y0[row] = Y0[row] * beta + tmpY[0][r] * alpha;
        Y1[row] = Y1[row] * beta + tmpY[1][r] * alpha;
        Y2[row] = Y2[row] * beta + tmpY[2][r] * alpha;
        Y3[row] = Y3[row] * beta + tmpY[3][r] * alpha;
    }
}

static alphasparseStatus_t
gemm_s_csr_col_chunk_unroll_col4_omp2(const float alpha, const internal_spmat mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *__restrict y, const ALPHA_INT ldy)
{
    ALPHA_INT chunk_num;
    chunk_s_t *chunks;
    csr_s_create_ccf(&chunks, &chunk_num, mat);
    // compute start

    ALPHA_INT eles[chunk_num];
    if (chunk_num > 0)
    {
        eles[0] = chunks[0].chuck_length * chunks[0].chuck_size;
    }
    for (ALPHA_INT i = 1; i < chunk_num; i++)
    {
        eles[i] = eles[i - 1] + chunks[i].chuck_length * chunks[i].chuck_size;
    }

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(eles, chunk_num, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT chunk_s = partition[tid];
        ALPHA_INT chunk_e = partition[tid + 1];
        ALPHA_INT col = 0;
        for (; col < columns - 3; col += 4)
        {
            const float *X0 = &x[index2(col, 0, ldx)];
            const float *X1 = &x[index2(col + 1, 0, ldx)];
            const float *X2 = &x[index2(col + 2, 0, ldx)];
            const float *X3 = &x[index2(col + 3, 0, ldx)];
            float *Y0 = &y[index2(col, 0, ldy)];
            float *Y1 = &y[index2(col + 1, 0, ldy)];
            float *Y2 = &y[index2(col + 2, 0, ldy)];
            float *Y3 = &y[index2(col + 3, 0, ldy)];
            for (ALPHA_INT chunk_id = chunk_s; chunk_id < chunk_e; ++chunk_id)
            {
                const chunk_s_t *chunk = &chunks[chunk_id];
                gemm_s_csr_col_chunk_unroll_col4_kernel(chunk, X0, X1, X2, X3, Y0, Y1, Y2, Y3, alpha, beta);
            }
        }
        for (; col < columns; col += 1)
        {
            const float *X0 = &x[index2(col, 0, ldx)];
            float *Y0 = &y[index2(col, 0, ldy)];
            for (ALPHA_INT chunk_id = chunk_s; chunk_id < chunk_e; ++chunk_id)
            {
                const chunk_s_t *chunk = &chunks[chunk_id];
                ALPHA_INT tmpC[CS] = {0};
                float tmpY[CS] = {0.f, 0.f, 0.f, 0.f};
                float tmpX[CS] = {0.f};
                float tmpV[CS] = {0.f};
                if (chunk->chuck_size == CS)
                {
                    for (ALPHA_INT l = 0, index = 0; l < chunk->chuck_length; ++l, index += CS)
                    {
                        tmpC[0] = chunk->cols[index];
                        tmpC[1] = chunk->cols[index + 1];
                        tmpC[2] = chunk->cols[index + 2];
                        tmpC[3] = chunk->cols[index + 3];
                        tmpV[0] = chunk->values[index];
                        tmpV[1] = chunk->values[index + 1];
                        tmpV[2] = chunk->values[index + 2];
                        tmpV[3] = chunk->values[index + 3];
                        tmpX[0] = X0[tmpC[0]];
                        tmpX[1] = X0[tmpC[1]];
                        tmpX[2] = X0[tmpC[2]];
                        tmpX[3] = X0[tmpC[3]];
                        tmpY[0] += tmpX[0] * tmpV[0];
                        tmpY[1] += tmpX[1] * tmpV[1];
                        tmpY[2] += tmpX[2] * tmpV[2];
                        tmpY[3] += tmpX[3] * tmpV[3];
                    }
                }
                else
                {
                    for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
                    {
                        for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                        {
                            tmpC[r] = chunk->cols[index2(l, r, chunk->chuck_size)];
                            tmpV[r] = chunk->values[index2(l, r, chunk->chuck_size)];
                            tmpX[r] = X0[tmpC[r]];
                            tmpY[r] += tmpX[r] * tmpV[r];
                        }
                    }
                }
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    ALPHA_INT row = chunk->rows[r];
                    Y0[row] = Y0[row] * beta + tmpY[r] * alpha;
                }
            }
        }
    }
    for (ALPHA_INT i = 0; i < chunk_num; i++)
    {
        alpha_free(chunks[i].cols);
        alpha_free(chunks[i].values);
    }
    alpha_free(chunks);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

void matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT c = 0; c < colX; ++c)
    {
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = X[index2(c, r, ldX)];
        }
    }
}
void matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < rowX; ++r)
    {
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = X[index2(r, c, ldX)];
        }
    }
}

alphasparseStatus_t
gemm_csr_col(const float alpha, const internal_spmat mat, const float *x, const ALPHA_INT columns, const ALPHA_INT ldx, const float beta, float *y, const ALPHA_INT ldy)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    if (num_threads <= 16)
    {
        ALPHA_INT sizeX = mat->cols * columns;
        ALPHA_INT ldX = columns;
        float *X = (float *)alpha_memalign(sizeX * sizeof(float), DEFAULT_ALIGNMENT);
        ALPHA_INT sizeY = mat->rows * columns;
        ALPHA_INT ldY = columns;
        float *Y = (float *)alpha_memalign(sizeY * sizeof(float), DEFAULT_ALIGNMENT);
        pack_matrix_col2row(mat->cols, columns, x, ldx, X, ldX);
        pack_matrix_col2row(mat->rows, columns, y, ldy, Y, ldY);
        gemm_csr_row<float>(alpha, mat, X, columns, ldX, beta, Y, ldY);
        pack_matrix_row2col(mat->rows, columns, Y, ldY, y, ldy);
        alpha_free(X);
        alpha_free(Y);
    }
    else
    {
        gemm_s_csr_col_chunk_unroll_col4_omp2(alpha, mat, x, columns, ldx, beta, y, ldy);
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define CS 4

typedef struct
{
    ALPHA_INT row;
    ALPHA_INT length;
} row_length_d_t;

int row_length_d_cmp(const row_length_d_t *a, const row_length_d_t *b)
{
    return a->length - b->length;
}

typedef struct
{
    ALPHA_INT chuck_size;
    ALPHA_INT chuck_length;
    ALPHA_INT rows[CS];
    ALPHA_INT * __restrict cols; // chuck_size * chuck_length
    double * __restrict values;
} chunk_d_t;

static void csr_d_create_ccf(chunk_d_t **chunks_p, ALPHA_INT *chunk_num_p, const internal_spmat mat)
{
    ALPHA_INT rowA = mat->rows;
    row_length_d_t *row_length = (row_length_d_t *)alpha_malloc(sizeof(row_length_d_t) * mat->rows);
    for (ALPHA_INT r = 0; r < rowA; ++r)
    {
        row_length[r].row = r;
        row_length[r].length = mat->row_data[r+1] - mat->row_data[r];
    }
    qsort(row_length, rowA, sizeof(row_length_d_t), (__compar_fn_t)row_length_d_cmp);
    ALPHA_INT bin_num = 0;
    ALPHA_INT last_len = 0;
    if (rowA > 0)
    {
        last_len = row_length[0].length;
        bin_num += 1;
    }
    for (ALPHA_INT i = 1; i < rowA; ++i)
    {
        if (last_len != row_length[i].length)
        {
            bin_num += 1;
            last_len = row_length[i].length;
        }
    }
    ALPHA_INT *bin_offset = (ALPHA_INT *)alpha_malloc(sizeof(ALPHA_INT) * (bin_num + 1));
    last_len = rowA > 0 ? row_length[0].length : 0;
    bin_offset[0] = 0;
    ALPHA_INT bin_index = 0;
    for (ALPHA_INT i = 1; i < rowA; ++i)
    {
        if (last_len != row_length[i].length)
        {
            bin_index += 1;
            last_len = row_length[i].length;
            bin_offset[bin_index] = i;
        }
    }
    bin_offset[bin_num] = rowA;
    ALPHA_INT chunk_num = 0;
    for (ALPHA_INT bin_id = 0; bin_id < bin_num; bin_id++)
    {
        ALPHA_INT bin_item_num = bin_offset[bin_id + 1] - bin_offset[bin_id];
        chunk_num += (bin_item_num + CS - 1) / CS;
    }
    chunk_d_t *chunks = (chunk_d_t *)alpha_malloc(sizeof(chunk_d_t) * chunk_num);
    ALPHA_INT chunk_index = 0;
    for (ALPHA_INT bin_id = 0; bin_id < bin_num; bin_id++)
    {
        ALPHA_INT bin_start = bin_offset[bin_id];
        ALPHA_INT bin_end = bin_offset[bin_id + 1];
        for (ALPHA_INT chunk_start = bin_start; chunk_start < bin_end; chunk_start += CS)
        {
            ALPHA_INT chunk_end = alpha_min(chunk_start + CS, bin_end);
            chunk_d_t *chunk = &chunks[chunk_index];
            chunk->chuck_size = chunk_end - chunk_start;
            chunk->chuck_length = row_length[bin_start].length;
            for (ALPHA_INT row_index = chunk_start; row_index < chunk_end; row_index++)
            {
                chunk->rows[row_index - chunk_start] = row_length[row_index].row;
            }
            chunk_index += 1;
        }
    }

    ALPHA_INT eles[chunk_num];
    if (chunk_num > 0)
    {
        eles[0] = chunks[0].chuck_length * chunks[0].chuck_size;
    }
    for (ALPHA_INT i = 1; i < chunk_num; i++)
    {
        eles[i] = eles[i - 1] + chunks[i].chuck_length * chunks[i].chuck_size;
    }

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(eles, chunk_num, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT chunk_s = partition[tid];
        ALPHA_INT chunk_e = partition[tid + 1];
        for (ALPHA_INT chunk_id = chunk_s; chunk_id < chunk_e; chunk_id++)
        {
            chunk_d_t *chunk = &chunks[chunk_id];
            chunk->cols = (ALPHA_INT *)alpha_malloc(chunk->chuck_size * chunk->chuck_length * sizeof(ALPHA_INT));
            chunk->values = (double *)alpha_malloc(chunk->chuck_size * chunk->chuck_length * sizeof(double));
            for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
            {
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    ALPHA_INT row = chunk->rows[r];
                    ALPHA_INT csr_index = mat->row_data[row] + l;
                    chunk->cols[index2(l, r, chunk->chuck_size)] = mat->col_data[csr_index];
                    chunk->values[index2(l, r, chunk->chuck_size)] = ((double *)mat->val_data)[csr_index];
                }
            }
        }
    }
    alpha_free(row_length);
    alpha_free(bin_offset);
    *chunks_p = chunks;
    *chunk_num_p = chunk_num;
}

void gemm_d_csr_col_chunk_unroll_col4_kernel(const chunk_d_t *chunk, const double *X0, const double *X1, const double *X2, const double *X3,
                                             double *__restrict Y0, double *__restrict Y1, double *__restrict Y2, double *__restrict Y3,
                                             const double alpha, const double beta)
{
    double tmpY[4][CS];
    double tmpX[4][CS] = {{0}};
    ALPHA_INT tmpC[CS] = {0};
    double tmpV[CS] = {0};
    memset(tmpY, '\0', 4 * CS * sizeof(double));
    if (chunk->chuck_size == CS)
    {
        for (ALPHA_INT l = 0, index = 0; l < chunk->chuck_length; ++l, index += CS)
        {
            tmpC[0] = chunk->cols[index];
            tmpC[1] = chunk->cols[index + 1];
            tmpC[2] = chunk->cols[index + 2];
            tmpC[3] = chunk->cols[index + 3];
            tmpV[0] = chunk->values[index];
            tmpV[1] = chunk->values[index + 1];
            tmpV[2] = chunk->values[index + 2];
            tmpV[3] = chunk->values[index + 3];
            tmpX[0][0] = X0[tmpC[0]];
            tmpX[0][1] = X0[tmpC[1]];
            tmpX[0][2] = X0[tmpC[2]];
            tmpX[0][3] = X0[tmpC[3]];
            tmpX[1][0] = X1[tmpC[0]];
            tmpX[1][1] = X1[tmpC[1]];
            tmpX[1][2] = X1[tmpC[2]];
            tmpX[1][3] = X1[tmpC[3]];
            tmpX[2][0] = X2[tmpC[0]];
            tmpX[2][1] = X2[tmpC[1]];
            tmpX[2][2] = X2[tmpC[2]];
            tmpX[2][3] = X2[tmpC[3]];
            tmpX[3][0] = X3[tmpC[0]];
            tmpX[3][1] = X3[tmpC[1]];
            tmpX[3][2] = X3[tmpC[2]];
            tmpX[3][3] = X3[tmpC[3]];
            tmpY[0][0] += tmpX[0][0] * tmpV[0];
            tmpY[0][1] += tmpX[0][1] * tmpV[1];
            tmpY[0][2] += tmpX[0][2] * tmpV[2];
            tmpY[0][3] += tmpX[0][3] * tmpV[3];
            tmpY[1][0] += tmpX[1][0] * tmpV[0];
            tmpY[1][1] += tmpX[1][1] * tmpV[1];
            tmpY[1][2] += tmpX[1][2] * tmpV[2];
            tmpY[1][3] += tmpX[1][3] * tmpV[3];
            tmpY[2][0] += tmpX[2][0] * tmpV[0];
            tmpY[2][1] += tmpX[2][1] * tmpV[1];
            tmpY[2][2] += tmpX[2][2] * tmpV[2];
            tmpY[2][3] += tmpX[2][3] * tmpV[3];
            tmpY[3][0] += tmpX[3][0] * tmpV[0];
            tmpY[3][1] += tmpX[3][1] * tmpV[1];
            tmpY[3][2] += tmpX[3][2] * tmpV[2];
            tmpY[3][3] += tmpX[3][3] * tmpV[3];
        }
    }
    else
    {
        for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
        {
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                tmpC[r] = chunk->cols[index2(l, r, chunk->chuck_size)];
                tmpV[r] = chunk->values[index2(l, r, chunk->chuck_size)];
            }
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                tmpX[0][r] = X0[tmpC[r]];
                tmpX[1][r] = X1[tmpC[r]];
                tmpX[2][r] = X2[tmpC[r]];
                tmpX[3][r] = X3[tmpC[r]];
            }
            for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
            {
                tmpY[0][r] += tmpX[0][r] * tmpV[r];
                tmpY[1][r] += tmpX[1][r] * tmpV[r];
                tmpY[2][r] += tmpX[2][r] * tmpV[r];
                tmpY[3][r] += tmpX[3][r] * tmpV[r];
            }
        }
    }
    for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
    {
        ALPHA_INT row = chunk->rows[r];
        Y0[row] = Y0[row] * beta + tmpY[0][r] * alpha;
        Y1[row] = Y1[row] * beta + tmpY[1][r] * alpha;
        Y2[row] = Y2[row] * beta + tmpY[2][r] * alpha;
        Y3[row] = Y3[row] * beta + tmpY[3][r] * alpha;
    }
}

alphasparseStatus_t
gemm_d_csr_col_chunk_unroll_col4_omp2(const double alpha, const internal_spmat mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *__restrict y, const ALPHA_INT ldy)
{
    ALPHA_INT chunk_num;
    chunk_d_t *chunks;
    csr_d_create_ccf(&chunks, &chunk_num, mat);
    // compute start

    ALPHA_INT eles[chunk_num];
    if (chunk_num > 0)
    {
        eles[0] = chunks[0].chuck_length * chunks[0].chuck_size;
    }
    for (ALPHA_INT i = 1; i < chunk_num; i++)
    {
        eles[i] = eles[i - 1] + chunks[i].chuck_length * chunks[i].chuck_size;
    }

    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(eles, chunk_num, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT chunk_s = partition[tid];
        ALPHA_INT chunk_e = partition[tid + 1];
        ALPHA_INT col = 0;
        for (; col < columns - 3; col += 4)
        {
            const double *X0 = &x[index2(col, 0, ldx)];
            const double *X1 = &x[index2(col + 1, 0, ldx)];
            const double *X2 = &x[index2(col + 2, 0, ldx)];
            const double *X3 = &x[index2(col + 3, 0, ldx)];
            double *Y0 = &y[index2(col, 0, ldy)];
            double *Y1 = &y[index2(col + 1, 0, ldy)];
            double *Y2 = &y[index2(col + 2, 0, ldy)];
            double *Y3 = &y[index2(col + 3, 0, ldy)];
            for (ALPHA_INT chunk_id = chunk_s; chunk_id < chunk_e; ++chunk_id)
            {
                const chunk_d_t *chunk = &chunks[chunk_id];
                gemm_d_csr_col_chunk_unroll_col4_kernel(chunk, X0, X1, X2, X3, Y0, Y1, Y2, Y3, alpha, beta);
            }
        }
        for (; col < columns; col += 1)
        {
            const double *X0 = &x[index2(col, 0, ldx)];
            double *Y0 = &y[index2(col, 0, ldy)];
            for (ALPHA_INT chunk_id = chunk_s; chunk_id < chunk_e; ++chunk_id)
            {
                const chunk_d_t *chunk = &chunks[chunk_id];
                ALPHA_INT tmpC[CS] = {0};
                double tmpY[CS] = {0, 0, 0, 0};
                double tmpX[CS] = {0};
                double tmpV[CS] = {0};
                if (chunk->chuck_size == CS)
                {
                    for (ALPHA_INT l = 0, index = 0; l < chunk->chuck_length; ++l, index += CS)
                    {
                        tmpC[0] = chunk->cols[index];
                        tmpC[1] = chunk->cols[index + 1];
                        tmpC[2] = chunk->cols[index + 2];
                        tmpC[3] = chunk->cols[index + 3];
                        tmpV[0] = chunk->values[index];
                        tmpV[1] = chunk->values[index + 1];
                        tmpV[2] = chunk->values[index + 2];
                        tmpV[3] = chunk->values[index + 3];
                        tmpX[0] = X0[tmpC[0]];
                        tmpX[1] = X0[tmpC[1]];
                        tmpX[2] = X0[tmpC[2]];
                        tmpX[3] = X0[tmpC[3]];
                        tmpY[0] += tmpX[0] * tmpV[0];
                        tmpY[1] += tmpX[1] * tmpV[1];
                        tmpY[2] += tmpX[2] * tmpV[2];
                        tmpY[3] += tmpX[3] * tmpV[3];
                    }
                }
                else
                {
                    for (ALPHA_INT l = 0; l < chunk->chuck_length; ++l)
                    {
                        for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                        {
                            tmpC[r] = chunk->cols[index2(l, r, chunk->chuck_size)];
                            tmpV[r] = chunk->values[index2(l, r, chunk->chuck_size)];
                            tmpX[r] = X0[tmpC[r]];
                            tmpY[r] += tmpX[r] * tmpV[r];
                        }
                    }
                }
                for (ALPHA_INT r = 0; r < chunk->chuck_size; ++r)
                {
                    ALPHA_INT row = chunk->rows[r];
                    Y0[row] = Y0[row] * beta + tmpY[r] * alpha;
                }
            }
        }
    }
    for (ALPHA_INT i = 0; i < chunk_num; i++)
    {
        alpha_free(chunks[i].cols);
        alpha_free(chunks[i].values);
    }
    alpha_free(chunks);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t gemm_csr_col(const double alpha, const internal_spmat mat, const double *x, const ALPHA_INT columns, const ALPHA_INT ldx, const double beta, double *y, const ALPHA_INT ldy)
{
    gemm_d_csr_col_chunk_unroll_col4_omp2(alpha, mat, x, columns, ldx, beta, y, ldy);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename J>
static void
gemm_csr_col_naive(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy, ALPHA_INT local_m_s, ALPHA_INT local_m_e)
{
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
    {
        for (ALPHA_INT cr = local_m_s; cr < local_m_e; ++cr)
        {
            J ctmp;
            alpha_setzero(ctmp);
            ALPHA_INT start = mat->row_data[cr];
            ALPHA_INT end = mat->row_data[cr+1];
            ctmp = vec_doti(end - start, &((J*)mat->val_data)[start], &mat->col_data[start], &x[index2(cc, 0, ldx)]);
            // alpha_mul(ctmp, alpha, ctmp);
            y[index2(cc, cr, ldy)] = alpha_mul(y[index2(cc, cr, ldy)], beta);
            y[index2(cc, cr, ldy)] = alpha_madd(ctmp, alpha, y[index2(cc, cr, ldy)]);
        }
    }
}

template <typename J>
static void
gemm_csr_col_omp(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT partition[num_threads + 1];
    balanced_partition_row_by_nnz(mat->row_data+1, m, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        ALPHA_INT tid = alpha_get_thread_id();
        ALPHA_INT local_m_s = partition[tid];
        ALPHA_INT local_m_e = partition[tid + 1];
        gemm_csr_col_naive(alpha, mat, x, columns, ldx, beta, y, ldy, local_m_s, local_m_e);
    }
}

template <typename J>
alphasparseStatus_t
gemm_csr_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT m = mat->rows;
    ALPHA_INT n = mat->cols;
    ALPHA_INT num_threads = alpha_get_thread_num();
    if (num_threads <= 20)
    {
        ALPHA_INT ldX = columns;
        ALPHA_INT ldY = columns;
        J *X_ = (J*)alpha_malloc(mat->cols * ldX * sizeof(J));
        J *Y_ = (J*)alpha_malloc(mat->rows * ldY * sizeof(J));

        pack_c2r(mat->cols, columns, x, ldx, X_, ldX);
        pack_c2r(mat->rows, columns, y, ldy, Y_, ldY);
        alphasparseStatus_t status = gemm_csr_row(alpha, mat, X_, columns, columns, beta, Y_, columns);
        pack_r2c(mat->rows, columns, Y_, ldY, y, ldy);
        alpha_free(X_);
        return status;
    }
    else{
        gemm_csr_col_omp<J>(alpha, mat, x, columns, ldx, beta, y, ldy);

    return ALPHA_SPARSE_STATUS_SUCCESS;
    }
}

#undef HEADER_PATH