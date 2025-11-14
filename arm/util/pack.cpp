#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/types.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef __aarch64__
#include <arm_neon.h>
#endif
void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(num_threads)
    // #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const float *xp0 = &X[index2(c, 0, ldX)];
        const float *xp1 = &X[index2(c + 1, 0, ldX)];
        const float *xp2 = &X[index2(c + 2, 0, ldX)];
        const float *xp3 = &X[index2(c + 3, 0, ldX)];
        float *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = colX4 < 0 ? 0 : colX4; c < colX; c += 1)
    {
        const float *xp0 = &X[index2(c, 0, ldX)];
        // #ifdef _OPENMP
        // #pragma omp parallel for num_threads(num_threads)
        // #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = 8 > alpha_get_thread_num() ? alpha_get_thread_num() : 8;
    ALPHA_INT rowX4 = rowX - 15;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < rowX4; r += 16)
    {
        const float *xp0 = &X[index2(r, 0, ldX)];
        const float *xp1 = &X[index2(r + 1, 0, ldX)];
        const float *xp2 = &X[index2(r + 2, 0, ldX)];
        const float *xp3 = &X[index2(r + 3, 0, ldX)];
        const float *xp4 = &X[index2(r + 4, 0, ldX)];
        const float *xp5 = &X[index2(r + 5, 0, ldX)];
        const float *xp6 = &X[index2(r + 6, 0, ldX)];
        const float *xp7 = &X[index2(r + 7, 0, ldX)];
        const float *xp8 = &X[index2(r + 8, 0, ldX)];
        const float *xp9 = &X[index2(r + 9, 0, ldX)];
        const float *xp10 = &X[index2(r + 10, 0, ldX)];
        const float *xp11 = &X[index2(r + 11, 0, ldX)];
        const float *xp12 = &X[index2(r + 12, 0, ldX)];
        const float *xp13 = &X[index2(r + 13, 0, ldX)];
        const float *xp14 = &X[index2(r + 14, 0, ldX)];
        const float *xp15 = &X[index2(r + 15, 0, ldX)];
        float *yp = &Y[r];
        float *yp1 = yp + ldY;
        float *yp2 = yp1 + ldY;
        float *yp3 = yp2 + ldY;
        ALPHA_INT colX4 = colX - 3;
        ALPHA_INT c = 0;
        for (; c < colX4; c += 4)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            *(yp + 8) = *(xp8++);
            *(yp + 9) = *(xp9++);
            *(yp + 10) = *(xp10++);
            *(yp + 11) = *(xp11++);
            *(yp + 12) = *(xp12++);
            *(yp + 13) = *(xp13++);
            *(yp + 14) = *(xp14++);
            *(yp + 15) = *(xp15++);
            yp += 4 * ldY;
            *yp1 = *(xp0++);
            *(yp1 + 1) = *(xp1++);
            *(yp1 + 2) = *(xp2++);
            *(yp1 + 3) = *(xp3++);
            *(yp1 + 4) = *(xp4++);
            *(yp1 + 5) = *(xp5++);
            *(yp1 + 6) = *(xp6++);
            *(yp1 + 7) = *(xp7++);
            *(yp1 + 8) = *(xp8++);
            *(yp1 + 9) = *(xp9++);
            *(yp1 + 10) = *(xp10++);
            *(yp1 + 11) = *(xp11++);
            *(yp1 + 12) = *(xp12++);
            *(yp1 + 13) = *(xp13++);
            *(yp1 + 14) = *(xp14++);
            *(yp1 + 15) = *(xp15++);
            yp1 += 4 * ldY;

            *yp2 = *(xp0++);
            *(yp2 + 1) = *(xp1++);
            *(yp2 + 2) = *(xp2++);
            *(yp2 + 3) = *(xp3++);
            *(yp2 + 4) = *(xp4++);
            *(yp2 + 5) = *(xp5++);
            *(yp2 + 6) = *(xp6++);
            *(yp2 + 7) = *(xp7++);
            *(yp2 + 8) = *(xp8++);
            *(yp2 + 9) = *(xp9++);
            *(yp2 + 10) = *(xp10++);
            *(yp2 + 11) = *(xp11++);
            *(yp2 + 12) = *(xp12++);
            *(yp2 + 13) = *(xp13++);
            *(yp2 + 14) = *(xp14++);
            *(yp2 + 15) = *(xp15++);
            yp2 += 4 * ldY;

            *yp3 = *(xp0++);
            *(yp3 + 1) = *(xp1++);
            *(yp3 + 2) = *(xp2++);
            *(yp3 + 3) = *(xp3++);
            *(yp3 + 4) = *(xp4++);
            *(yp3 + 5) = *(xp5++);
            *(yp3 + 6) = *(xp6++);
            *(yp3 + 7) = *(xp7++);
            *(yp3 + 8) = *(xp8++);
            *(yp3 + 9) = *(xp9++);
            *(yp3 + 10) = *(xp10++);
            *(yp3 + 11) = *(xp11++);
            *(yp3 + 12) = *(xp12++);
            *(yp3 + 13) = *(xp13++);
            *(yp3 + 14) = *(xp14++);
            *(yp3 + 15) = *(xp15++);
            yp3 += 4 * ldY;
        }
        for (; c < colX; c++)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            *(yp + 8) = *(xp8++);
            *(yp + 9) = *(xp9++);
            *(yp + 10) = *(xp10++);
            *(yp + 11) = *(xp11++);
            *(yp + 12) = *(xp12++);
            *(yp + 13) = *(xp13++);
            *(yp + 14) = *(xp14++);
            *(yp + 15) = *(xp15++);
            yp += ldY;
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const float *xp0 = &X[index2(r, 0, ldX)];
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(num_threads)
    // #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const double *xp0 = &X[index2(c, 0, ldX)];
        const double *xp1 = &X[index2(c + 1, 0, ldX)];
        const double *xp2 = &X[index2(c + 2, 0, ldX)];
        const double *xp3 = &X[index2(c + 3, 0, ldX)];
        double *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = colX4 < 0 ? 0 : colX4; c < colX; c += 1)
    {
        const double *xp0 = &X[index2(c, 0, ldX)];
        // #ifdef _OPENMP
        // #pragma omp parallel for num_threads(num_threads)
        // #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = 8 > alpha_get_thread_num() ? alpha_get_thread_num() : 8;
    ALPHA_INT rowX4 = rowX - 7;
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
    for (ALPHA_INT r = 0; r < rowX4; r += 8)
    {
        const double *xp0 = &X[index2(r, 0, ldX)];
        const double *xp1 = &X[index2(r + 1, 0, ldX)];
        const double *xp2 = &X[index2(r + 2, 0, ldX)];
        const double *xp3 = &X[index2(r + 3, 0, ldX)];
        const double *xp4 = &X[index2(r + 4, 0, ldX)];
        const double *xp5 = &X[index2(r + 5, 0, ldX)];
        const double *xp6 = &X[index2(r + 6, 0, ldX)];
        const double *xp7 = &X[index2(r + 7, 0, ldX)];
        // const double *xp8 = &X[index2(r + 8, 0, ldX)];
        // const double *xp9 = &X[index2(r + 9, 0, ldX)];
        // const double *xp10 = &X[index2(r + 10, 0, ldX)];
        // const double *xp11 = &X[index2(r + 11, 0, ldX)];
        // const double *xp12 = &X[index2(r + 12, 0, ldX)];
        // const double *xp13 = &X[index2(r + 13, 0, ldX)];
        // const double *xp14 = &X[index2(r + 14, 0, ldX)];
        // const double *xp15 = &X[index2(r + 15, 0, ldX)];
        double *yp = &Y[r];
        double *yp1 = yp + ldY;
        double *yp2 = yp1 + ldY;
        double *yp3 = yp2 + ldY;
        ALPHA_INT colX4 = colX - 3;
        ALPHA_INT c = 0;
        for (; c < colX4; c += 4)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            // *(yp + 8) = *(xp8++);
            // *(yp + 9) = *(xp9++);
            // *(yp + 10) = *(xp10++);
            // *(yp + 11) = *(xp11++);
            // *(yp + 12) = *(xp12++);
            // *(yp + 13) = *(xp13++);
            // *(yp + 14) = *(xp14++);
            // *(yp + 15) = *(xp15++);
            yp += 4 * ldY;
            *yp1 = *(xp0++);
            *(yp1 + 1) = *(xp1++);
            *(yp1 + 2) = *(xp2++);
            *(yp1 + 3) = *(xp3++);
            *(yp1 + 4) = *(xp4++);
            *(yp1 + 5) = *(xp5++);
            *(yp1 + 6) = *(xp6++);
            *(yp1 + 7) = *(xp7++);
            // *(yp1 + 8) = *(xp8++);
            // *(yp1 + 9) = *(xp9++);
            // *(yp1 + 10) = *(xp10++);
            // *(yp1 + 11) = *(xp11++);
            // *(yp1 + 12) = *(xp12++);
            // *(yp1 + 13) = *(xp13++);
            // *(yp1 + 14) = *(xp14++);
            // *(yp1 + 15) = *(xp15++);
            yp1 += 4 * ldY;

            *yp2 = *(xp0++);
            *(yp2 + 1) = *(xp1++);
            *(yp2 + 2) = *(xp2++);
            *(yp2 + 3) = *(xp3++);
            *(yp2 + 4) = *(xp4++);
            *(yp2 + 5) = *(xp5++);
            *(yp2 + 6) = *(xp6++);
            *(yp2 + 7) = *(xp7++);
            // *(yp2 + 8) = *(xp8++);
            // *(yp2 + 9) = *(xp9++);
            // *(yp2 + 10) = *(xp10++);
            // *(yp2 + 11) = *(xp11++);
            // *(yp2 + 12) = *(xp12++);
            // *(yp2 + 13) = *(xp13++);
            // *(yp2 + 14) = *(xp14++);
            // *(yp2 + 15) = *(xp15++);
            yp2 += 4 * ldY;

            *yp3 = *(xp0++);
            *(yp3 + 1) = *(xp1++);
            *(yp3 + 2) = *(xp2++);
            *(yp3 + 3) = *(xp3++);
            *(yp3 + 4) = *(xp4++);
            *(yp3 + 5) = *(xp5++);
            *(yp3 + 6) = *(xp6++);
            *(yp3 + 7) = *(xp7++);
            // *(yp3 + 8) = *(xp8++);
            // *(yp3 + 9) = *(xp9++);
            // *(yp3 + 10) = *(xp10++);
            // *(yp3 + 11) = *(xp11++);
            // *(yp3 + 12) = *(xp12++);
            // *(yp3 + 13) = *(xp13++);
            // *(yp3 + 14) = *(xp14++);
            // *(yp3 + 15) = *(xp15++);
            yp3 += 4 * ldY;
        }
        for (; c < colX; c++)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            *(yp + 4) = *(xp4++);
            *(yp + 5) = *(xp5++);
            *(yp + 6) = *(xp6++);
            *(yp + 7) = *(xp7++);
            // *(yp + 8) = *(xp8++);
            // *(yp + 9) = *(xp9++);
            // *(yp + 10) = *(xp10++);
            // *(yp + 11) = *(xp11++);
            // *(yp + 12) = *(xp12++);
            // *(yp + 13) = *(xp13++);
            // *(yp + 14) = *(xp14++);
            // *(yp + 15) = *(xp15++);
            yp += ldY;
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const double *xp0 = &X[index2(r, 0, ldX)];
#ifdef _OPENMP
#pragma omp parallel for num_threads(num_threads)
#endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(num_threads)
    // #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(c, 0, ldX)];
        const ALPHA_Complex8 *xp1 = &X[index2(c + 1, 0, ldX)];
        const ALPHA_Complex8 *xp2 = &X[index2(c + 2, 0, ldX)];
        const ALPHA_Complex8 *xp3 = &X[index2(c + 3, 0, ldX)];
        ALPHA_Complex8 *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = (colX4 < 0 ? 0 : colX4); c < colX; c += 1)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(c, 0, ldX)];
        // #ifdef _OPENMP
        // #pragma omp parallel for num_threads(num_threads)
        // #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT rowX4 = rowX - 3;
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(num_threads)
    // #endif
    for (ALPHA_INT r = 0; r < rowX4; r += 4)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(r, 0, ldX)];
        const ALPHA_Complex8 *xp1 = &X[index2(r + 1, 0, ldX)];
        const ALPHA_Complex8 *xp2 = &X[index2(r + 2, 0, ldX)];
        const ALPHA_Complex8 *xp3 = &X[index2(r + 3, 0, ldX)];
        ALPHA_Complex8 *yp = &Y[r];
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const ALPHA_Complex8 *xp0 = &X[index2(r, 0, ldX)];
        // #ifdef _OPENMP
        // #pragma omp parallel for num_threads(num_threads)
        // #endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_matrix_col2row(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT colX4 = colX - 3;
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(num_threads)
    // #endif
    for (ALPHA_INT c = 0; c < colX4; c += 4)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(c, 0, ldX)];
        const ALPHA_Complex16 *xp1 = &X[index2(c + 1, 0, ldX)];
        const ALPHA_Complex16 *xp2 = &X[index2(c + 2, 0, ldX)];
        const ALPHA_Complex16 *xp3 = &X[index2(c + 3, 0, ldX)];
        ALPHA_Complex16 *yp = &Y[c];
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT c = colX4 < 0 ? 0 : colX4; c < colX; c += 1)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(c, 0, ldX)];
        // #ifdef _OPENMP
        // #pragma omp parallel for num_threads(num_threads)
        // #endif
        for (ALPHA_INT r = 0; r < rowX; ++r)
        {
            Y[index2(r, c, ldY)] = xp0[r];
        }
    }
}

void pack_matrix_row2col(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    ALPHA_INT rowX4 = rowX - 3;
    // #ifdef _OPENMP
    // #pragma omp parallel for num_threads(num_threads)
    // #endif
    for (ALPHA_INT r = 0; r < rowX4; r += 4)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(r, 0, ldX)];
        const ALPHA_Complex16 *xp1 = &X[index2(r + 1, 0, ldX)];
        const ALPHA_Complex16 *xp2 = &X[index2(r + 2, 0, ldX)];
        const ALPHA_Complex16 *xp3 = &X[index2(r + 3, 0, ldX)];
        ALPHA_Complex16 *yp = &Y[r];
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            *yp = *(xp0++);
            *(yp + 1) = *(xp1++);
            *(yp + 2) = *(xp2++);
            *(yp + 3) = *(xp3++);
            yp += ldY;
        }
    }
    for (ALPHA_INT r = rowX4 < 0 ? 0 : rowX4; r < rowX; r += 1)
    {
        const ALPHA_Complex16 *xp0 = &X[index2(r, 0, ldX)];
        // #ifdef _OPENMP
        // #pragma omp parallel for num_threads(num_threads)
        // #endif
        for (ALPHA_INT c = 0; c < colX; ++c)
        {
            Y[index2(c, r, ldY)] = xp0[c];
        }
    }
}

void pack_r2cerial(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT row4 = (rowX >> 2) << 2;
    ALPHA_INT col4 = (rowX >> 2) << 2;

    ALPHA_INT r = 0, c = 0;
#ifdef __aarch64__
    float32x4_t v_row0, v_row1, v_row2, v_row3; // input
    float32x4_t T0, T1, T2, T3;
    float64x2_t TT0, TT1, TT2, TT3;
    float64x2_t v_col0, v_col1, v_col2, v_col3; // output
#endif
    for (c = 0; c < col4; c += 4)
    {

        float *y_0 = (float *)(Y + ldY * (c + 0));
        float *y_1 = (float *)(Y + ldY * (c + 1));
        float *y_2 = (float *)(Y + ldY * (c + 2));
        float *y_3 = (float *)(Y + ldY * (c + 3));
        for (r = 0; r < row4; r += 4)
        {
            const float *x_0 = X + ldX * (r + 0) + c;
            const float *x_1 = X + ldX * (r + 1) + c;
            const float *x_2 = X + ldX * (r + 2) + c;
            const float *x_3 = X + ldX * (r + 3) + c;
#ifdef __aarch64__
            v_row0 = vld1q_f32(x_0);
            v_row1 = vld1q_f32(x_1);
            v_row2 = vld1q_f32(x_2);
            v_row3 = vld1q_f32(x_3);

            T0 = vzip1q_f32(v_row0, v_row1);
            T1 = vzip2q_f32(v_row0, v_row1);
            T2 = vzip1q_f32(v_row2, v_row3);
            T3 = vzip2q_f32(v_row2, v_row3);

            TT0 = vreinterpretq_f64_f32(T0);
            TT1 = vreinterpretq_f64_f32(T1);
            TT2 = vreinterpretq_f64_f32(T2);
            TT3 = vreinterpretq_f64_f32(T3);

            v_col0 = vzip1q_f64(TT0, TT2);
            v_col1 = vzip2q_f64(TT0, TT2);
            v_col2 = vzip1q_f64(TT1, TT3);
            v_col3 = vzip2q_f64(TT1, TT3);

            vst1q_f64((double *)(y_0 + r), v_col0);
            vst1q_f64((double *)(y_1 + r), v_col1);
            vst1q_f64((double *)(y_2 + r), v_col2);
            vst1q_f64((double *)(y_3 + r), v_col3);
#else
            y_0[r + 0] = x_0[0];
            y_0[r + 1] = x_1[0];
            y_0[r + 2] = x_2[0];
            y_0[r + 3] = x_3[0];

            y_1[r + 0] = x_0[1];
            y_1[r + 1] = x_1[1];
            y_1[r + 2] = x_2[1];
            y_1[r + 3] = x_3[1];

            y_2[r + 0] = x_0[2];
            y_2[r + 1] = x_1[2];
            y_2[r + 2] = x_2[2];
            y_2[r + 3] = x_3[2];

            y_3[r + 0] = x_0[3];
            y_3[r + 1] = x_1[3];
            y_3[r + 2] = x_2[3];
            y_3[r + 3] = x_3[3];
#endif
        }
        for (; r < rowX; r++)
        {
            const float *x_0 = X + ldX * (r + 0) + c;
            y_0[r + 0] = x_0[0];
            y_1[r + 0] = x_0[1];
            y_2[r + 0] = x_0[2];
            y_3[r + 0] = x_0[3];
        }
    }
    for (; c < colX; c++)
    {
        float *y_0 = (float *)(Y + ldY * (c + 0));
        for (r = 0; r < rowX; r++)
        {
            const float *x_0 = X + ldX * (r + 0) + c;
            y_0[r + 0] = x_0[0];
        }
    }
}

//equivalent to transposing of a row-major dense matrix
void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
// #pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float32x4_t v_row0, v_row1, v_row2, v_row3; // input
        float32x4_t T0, T1, T2, T3;
        float64x2_t TT0, TT1, TT2, TT3;
        float64x2_t v_col0, v_col1, v_col2, v_col3; // output
#endif
#ifdef _OPENMP
// #pragma omp for
#endif
        for (ALPHA_INT cc = 0; cc < colX; cc += NPERCLF32)
        // ALPHA_INT cc = 0;
        {
            const ALPHA_INT c_up = alpha_min(cc + NPERCLF32, colX);
            const ALPHA_INT c_up4 = c_up - 3;
            ALPHA_INT c;
            for (ALPHA_INT rr = 0; rr < rowX; rr += NPERCLF32)
            {
                ALPHA_INT r;
                const ALPHA_INT r_up = alpha_min(rr + NPERCLF32, rowX);
                const ALPHA_INT r_up4 = r_up - 3;
                for (c = cc; c < c_up4; c += 4)
                {
                    float *y_0 = (float *)(Y + ldY * (c + 0));
                    float *y_1 = (float *)(Y + ldY * (c + 1));
                    float *y_2 = (float *)(Y + ldY * (c + 2));
                    float *y_3 = (float *)(Y + ldY * (c + 3));
                    for (r = rr; r < r_up4; r += 4)
                    {
                        const float *x_0 = X + ldX * (r + 0) + c;
                        const float *x_1 = X + ldX * (r + 1) + c;
                        const float *x_2 = X + ldX * (r + 2) + c;
                        const float *x_3 = X + ldX * (r + 3) + c;
#ifdef __aarch64__
                        v_row0 = vld1q_f32(x_0);
                        v_row1 = vld1q_f32(x_1);
                        v_row2 = vld1q_f32(x_2);
                        v_row3 = vld1q_f32(x_3);

                        T0 = vzip1q_f32(v_row0, v_row1);
                        T1 = vzip2q_f32(v_row0, v_row1);
                        T2 = vzip1q_f32(v_row2, v_row3);
                        T3 = vzip2q_f32(v_row2, v_row3);

                        TT0 = vreinterpretq_f64_f32(T0);
                        TT1 = vreinterpretq_f64_f32(T1);
                        TT2 = vreinterpretq_f64_f32(T2);
                        TT3 = vreinterpretq_f64_f32(T3);

                        v_col0 = vzip1q_f64(TT0, TT2);
                        v_col1 = vzip2q_f64(TT0, TT2);
                        v_col2 = vzip1q_f64(TT1, TT3);
                        v_col3 = vzip2q_f64(TT1, TT3);

                        vst1q_f64((double *)(y_0 + r), v_col0);
                        vst1q_f64((double *)(y_1 + r), v_col1);
                        vst1q_f64((double *)(y_2 + r), v_col2);
                        vst1q_f64((double *)(y_3 + r), v_col3);
#else
                        y_0[r + 0] = x_0[0];
                        y_0[r + 1] = x_1[0];
                        y_0[r + 2] = x_2[0];
                        y_0[r + 3] = x_3[0];

                        y_1[r + 0] = x_0[1];
                        y_1[r + 1] = x_1[1];
                        y_1[r + 2] = x_2[1];
                        y_1[r + 3] = x_3[1];

                        y_2[r + 0] = x_0[2];
                        y_2[r + 1] = x_1[2];
                        y_2[r + 2] = x_2[2];
                        y_2[r + 3] = x_3[2];

                        y_3[r + 0] = x_0[3];
                        y_3[r + 1] = x_1[3];
                        y_3[r + 2] = x_2[3];
                        y_3[r + 3] = x_3[3];
#endif
                    }
                    for (; r < r_up; r++)
                    {
                        const float *x_0 = X + ldX * (r + 0) + c;
                        y_0[r + 0] = x_0[0];
                        y_1[r + 0] = x_0[1];
                        y_2[r + 0] = x_0[2];
                        y_3[r + 0] = x_0[3];
                    }
                }
                for (; c < c_up; c++)
                {
                    float *y_0 = (float *)(Y + ldY * (c + 0));
                    for (r = rr; r < r_up; r++)
                    {
                        const float *x_0 = X + ldX * (r + 0) + c;
                        y_0[r + 0] = x_0[0];
                    }
                }
            }
        }
    }
}
//equivalent to transposing of a col-major dense matrix
void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
    // for (ALPHA_INT r = 0; r < rowX; r++)    
    // {
    //     for (ALPHA_INT c = 0; c < colX; c++)
    //     {
    //         printf("%f ", X[c * ldX + r]);
    //     }
    //     printf("\n");
    // }
    // printf("after trans\n");
#ifdef _OPENMP
// #pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float32x4_t v_row0, v_row1, v_row2, v_row3; // input
        float32x4_t T0, T1, T2, T3;
        float64x2_t TT0, TT1, TT2, TT3;
        float64x2_t v_col0, v_col1, v_col2, v_col3; // output
#endif
#ifdef _OPENMP
// #pragma omp for
#endif
        for (ALPHA_INT rr = 0; rr < rowX; rr += NPERCLF32)
        {
            ALPHA_INT r;
            const ALPHA_INT r_up = alpha_min(rr + NPERCLF32, rowX);
            const ALPHA_INT r_up4 = r_up - 3;
            for (ALPHA_INT cc = 0; cc < colX; cc += NPERCLF32)
            {
                const ALPHA_INT c_up = alpha_min(cc + NPERCLF32, colX);
                const ALPHA_INT c_up4 = c_up - 3;
                ALPHA_INT c;

                for (r = rr; r < r_up4; r += 4)
                {
                    float *y_0 = (float *)(Y + ldY * (r + 0));
                    float *y_1 = (float *)(Y + ldY * (r + 1));
                    float *y_2 = (float *)(Y + ldY * (r + 2));
                    float *y_3 = (float *)(Y + ldY * (r + 3));
                    for (c = cc; c < c_up4; c += 4)
                    {
                        const float *x_0 = X + ldX * (c + 0) + r;
                        const float *x_1 = X + ldX * (c + 1) + r;
                        const float *x_2 = X + ldX * (c + 2) + r;
                        const float *x_3 = X + ldX * (c + 3) + r;
#ifdef __aarch64__
                        v_row0 = vld1q_f32(x_0);
                        v_row1 = vld1q_f32(x_1);
                        v_row2 = vld1q_f32(x_2);
                        v_row3 = vld1q_f32(x_3);

                        T0 = vzip1q_f32(v_row0, v_row1);
                        T1 = vzip2q_f32(v_row0, v_row1);
                        T2 = vzip1q_f32(v_row2, v_row3);
                        T3 = vzip2q_f32(v_row2, v_row3);

                        TT0 = vreinterpretq_f64_f32(T0);
                        TT1 = vreinterpretq_f64_f32(T1);
                        TT2 = vreinterpretq_f64_f32(T2);
                        TT3 = vreinterpretq_f64_f32(T3);

                        v_col0 = vzip1q_f64(TT0, TT2);
                        v_col1 = vzip2q_f64(TT0, TT2);
                        v_col2 = vzip1q_f64(TT1, TT3);
                        v_col3 = vzip2q_f64(TT1, TT3);

                        vst1q_f64((double *)(y_0 + c), v_col0);
                        vst1q_f64((double *)(y_1 + c), v_col1);
                        vst1q_f64((double *)(y_2 + c), v_col2);
                        vst1q_f64((double *)(y_3 + c), v_col3);
#else
                        y_0[c + 0] = x_0[0];
                        y_0[c + 1] = x_1[0];
                        y_0[c + 2] = x_2[0];
                        y_0[c + 3] = x_3[0];

                        y_1[c + 0] = x_0[1];
                        y_1[c + 1] = x_1[1];
                        y_1[c + 2] = x_2[1];
                        y_1[c + 3] = x_3[1];

                        y_2[c + 0] = x_0[2];
                        y_2[c + 1] = x_1[2];
                        y_2[c + 2] = x_2[2];
                        y_2[c + 3] = x_3[2];

                        y_3[c + 0] = x_0[3];
                        y_3[c + 1] = x_1[3];
                        y_3[c + 2] = x_2[3];
                        y_3[c + 3] = x_3[3];
#endif
                    }
                    for (; c < c_up; c++)
                    {
                        const float *x_0 = X + ldX * (c + 0) + r;
                        y_0[c + 0] = x_0[0];
                        y_1[c + 0] = x_0[1];
                        y_2[c + 0] = x_0[2];
                        y_3[c + 0] = x_0[3];
                    }
                }
                for (; r < r_up; r++)
                {
                    float *y_0 = (float *)(Y + ldY * (r + 0));
                    for (c = cc; c < c_up; c++)
                    {
                        const float *x_0 = X + ldX * (c + 0) + r;
                        y_0[c + 0] = x_0[0];
                    }
                }
            }
        }
    }
    // for (ALPHA_INT r = 0; r < rowX; r++)
    // {
    //     for (ALPHA_INT c = 0; c < colX; c++)
    //     {
    //         printf("%f ", Y[c * ldX + r]);
    //     }
    //     printf("\n");
    // }
}
//memcpy
void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const float *X, const ALPHA_INT ldX, float *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float32x4_t v_row0, v_row1, v_row2, v_row3; //
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (ALPHA_INT r = 0; r < rowX; r++)
        {
            ALPHA_INT c = 0;
            //NPERCLF32 = 16 ; 512 / 128 = 4, thus need 4 register
            for (; c < colX - NPERCLF32 + 1; c += NPERCLF32)
            {
                const float *src = X + ldX * r + c;
                float *dst = Y + ldY * r + c;
#ifdef __aarch64__
                v_row0 = vld1q_f32(src + 0);
                v_row1 = vld1q_f32(src + 4);
                v_row2 = vld1q_f32(src + 8);
                v_row3 = vld1q_f32(src + 12);
                vst1q_f32(dst + 0, v_row0);
                vst1q_f32(dst + 4, v_row1);
                vst1q_f32(dst + 8, v_row2);
                vst1q_f32(dst + 12, v_row3);
#else
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
                dst[4] = src[4];
                dst[5] = src[5];
                dst[6] = src[6];
                dst[7] = src[7];
                dst[8] = src[8];
                dst[9] = src[9];
                dst[10] = src[10];
                dst[11] = src[11];
                dst[12] = src[12];
                dst[13] = src[13];
                dst[14] = src[14];
                dst[15] = src[15];
#endif
            }
            for (; c < colX - 3; c += 4)
            {
                const float *src = X + ldX * r + c;
                float *dst = Y + ldY * r + c;
#ifdef __aarch64__
                v_row0 = vld1q_f32(src + 0);
                vst1q_f32(dst + 0, v_row0);
#else
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
#endif
            }
            for (; c < colX; c++)
            {
                const float *src = X + ldX * r + c;
                float *dst = Y + ldY * r + c;
                dst[0] = src[0];
            }
        }
    }
}

void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();

#ifdef _OPENMP
// #pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float64x2_t v_row00, v_row10, v_row20, v_row30; // input
        float64x2_t v_row01, v_row11, v_row21, v_row31; // input
        float64x2_t T0, T1, T2, T3;
#endif
#ifdef _OPENMP
// #pragma omp for
#endif
        for (ALPHA_INT cc = 0; cc < colX; cc += NPERCLF64)
        {
            const ALPHA_INT c_up = alpha_min(cc + NPERCLF64, colX);
            const ALPHA_INT c_up4 = c_up - 3;
            ALPHA_INT c;
            for (ALPHA_INT rr = 0; rr < rowX; rr += NPERCLF64)
            {
                ALPHA_INT r;
                const ALPHA_INT r_up = alpha_min(rr + NPERCLF64, rowX);
                const ALPHA_INT r_up4 = r_up - 3;
                for (c = cc; c < c_up4; c += 4)
                {
                    double *y_0 = (double *)(Y + ldY * (c + 0));
                    double *y_1 = (double *)(Y + ldY * (c + 1));
                    double *y_2 = (double *)(Y + ldY * (c + 2));
                    double *y_3 = (double *)(Y + ldY * (c + 3));
                    for (r = rr; r < r_up4; r += 4)
                    {
                        const double *x_0 = X + ldX * (r + 0) + c;
                        const double *x_1 = X + ldX * (r + 1) + c;
                        const double *x_2 = X + ldX * (r + 2) + c;
                        const double *x_3 = X + ldX * (r + 3) + c;
#ifdef __aarch64__
                        v_row00 = vld1q_f64(x_0);
                        v_row10 = vld1q_f64(x_1);
                        v_row20 = vld1q_f64(x_2);
                        v_row30 = vld1q_f64(x_3);

                        T0 = vzip1q_f64(v_row00, v_row10);
                        T1 = vzip2q_f64(v_row00, v_row10);
                        T2 = vzip1q_f64(v_row20, v_row30);
                        T3 = vzip2q_f64(v_row20, v_row30);

                        vst1q_f64((double *)(y_0 + r), T0);
                        vst1q_f64((double *)(y_1 + r), T1);
                        vst1q_f64((double *)(y_0 + r + 2), T2);
                        vst1q_f64((double *)(y_1 + r + 2), T3);

                        v_row01 = vld1q_f64(x_0 + 2);
                        v_row11 = vld1q_f64(x_1 + 2);
                        v_row21 = vld1q_f64(x_2 + 2);
                        v_row31 = vld1q_f64(x_3 + 2);

                        T0 = vzip1q_f64(v_row01, v_row11);
                        T1 = vzip2q_f64(v_row01, v_row11);
                        T2 = vzip1q_f64(v_row21, v_row31);
                        T3 = vzip2q_f64(v_row21, v_row31);

                        vst1q_f64((double *)(y_2 + r), T0);
                        vst1q_f64((double *)(y_3 + r), T1);
                        vst1q_f64((double *)(y_2 + r + 2), T2);
                        vst1q_f64((double *)(y_3 + r + 2), T3);

#else
                        y_0[r + 0] = x_0[0];
                        y_0[r + 1] = x_1[0];
                        y_0[r + 2] = x_2[0];
                        y_0[r + 3] = x_3[0];

                        y_1[r + 0] = x_0[1];
                        y_1[r + 1] = x_1[1];
                        y_1[r + 2] = x_2[1];
                        y_1[r + 3] = x_3[1];

                        y_2[r + 0] = x_0[2];
                        y_2[r + 1] = x_1[2];
                        y_2[r + 2] = x_2[2];
                        y_2[r + 3] = x_3[2];

                        y_3[r + 0] = x_0[3];
                        y_3[r + 1] = x_1[3];
                        y_3[r + 2] = x_2[3];
                        y_3[r + 3] = x_3[3];
#endif
                    }
                    for (; r < r_up; r++)
                    {
                        const double *x_0 = X + ldX * (r + 0) + c;
                        y_0[r + 0] = x_0[0];
                        y_1[r + 0] = x_0[1];
                        y_2[r + 0] = x_0[2];
                        y_3[r + 0] = x_0[3];
                    }
                }
                for (; c < c_up; c++)
                {
                    double *y_0 = (double *)(Y + ldY * (c + 0));
                    for (r = rr; r < r_up; r++)
                    {
                        const double *x_0 = X + ldX * (r + 0) + c;
                        y_0[r + 0] = x_0[0];
                    }
                }
            }
        }
    }
}

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
// #pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float64x2_t v_row00, v_row10, v_row20, v_row30; // input
        float64x2_t v_row01, v_row11, v_row21, v_row31; // input
        float64x2_t T0, T1, T2, T3;
#endif
#ifdef _OPENMP
// #pragma omp for
#endif
        for (ALPHA_INT rr = 0; rr < rowX; rr += NPERCLF64)
        {
            ALPHA_INT r;
            const ALPHA_INT r_up = alpha_min(rr + NPERCLF64, rowX);
            const ALPHA_INT r_up4 = r_up - 3;
            for (ALPHA_INT cc = 0; cc < colX; cc += NPERCLF64)
            {
                const ALPHA_INT c_up = alpha_min(cc + NPERCLF64, colX);
                const ALPHA_INT c_up4 = c_up - 3;
                ALPHA_INT c;

                for (r = rr; r < r_up4; r += 4)
                {
                    double *y_0 = (double *)(Y + ldY * (r + 0));
                    double *y_1 = (double *)(Y + ldY * (r + 1));
                    double *y_2 = (double *)(Y + ldY * (r + 2));
                    double *y_3 = (double *)(Y + ldY * (r + 3));
                    for (c = cc; c < c_up4; c += 4)
                    {
                        const double *x_0 = X + ldX * (c + 0) + r;
                        const double *x_1 = X + ldX * (c + 1) + r;
                        const double *x_2 = X + ldX * (c + 2) + r;
                        const double *x_3 = X + ldX * (c + 3) + r;
#ifdef __aarch64__
                        v_row00 = vld1q_f64(x_0);
                        v_row10 = vld1q_f64(x_1);
                        v_row20 = vld1q_f64(x_2);
                        v_row30 = vld1q_f64(x_3);

                        T0 = vzip1q_f64(v_row00, v_row10);
                        T1 = vzip2q_f64(v_row00, v_row10);
                        T2 = vzip1q_f64(v_row20, v_row30);
                        T3 = vzip2q_f64(v_row20, v_row30);

                        vst1q_f64((double *)(y_0 + c), T0);
                        vst1q_f64((double *)(y_1 + c), T1);
                        vst1q_f64((double *)(y_0 + c + 2), T2);
                        vst1q_f64((double *)(y_1 + c + 2), T3);

                        v_row01 = vld1q_f64(x_0 + 2);
                        v_row11 = vld1q_f64(x_1 + 2);
                        v_row21 = vld1q_f64(x_2 + 2);
                        v_row31 = vld1q_f64(x_3 + 2);

                        T0 = vzip1q_f64(v_row01, v_row11);
                        T1 = vzip2q_f64(v_row01, v_row11);
                        T2 = vzip1q_f64(v_row21, v_row31);
                        T3 = vzip2q_f64(v_row21, v_row31);

                        vst1q_f64((double *)(y_2 + c), T0);
                        vst1q_f64((double *)(y_3 + c), T1);
                        vst1q_f64((double *)(y_2 + c + 2), T2);
                        vst1q_f64((double *)(y_3 + c + 2), T3);
#else
                        y_0[c + 0] = x_0[0];
                        y_0[c + 1] = x_1[0];
                        y_0[c + 2] = x_2[0];
                        y_0[c + 3] = x_3[0];

                        y_1[c + 0] = x_0[1];
                        y_1[c + 1] = x_1[1];
                        y_1[c + 2] = x_2[1];
                        y_1[c + 3] = x_3[1];

                        y_2[c + 0] = x_0[2];
                        y_2[c + 1] = x_1[2];
                        y_2[c + 2] = x_2[2];
                        y_2[c + 3] = x_3[2];

                        y_3[c + 0] = x_0[3];
                        y_3[c + 1] = x_1[3];
                        y_3[c + 2] = x_2[3];
                        y_3[c + 3] = x_3[3];
#endif
                    }
                    for (; c < c_up; c++)
                    {
                        const double *x_0 = X + ldX * (c + 0) + r;
                        y_0[c + 0] = x_0[0];
                        y_1[c + 0] = x_0[1];
                        y_2[c + 0] = x_0[2];
                        y_3[c + 0] = x_0[3];
                    }
                }
                for (; r < r_up; r++)
                {
                    double *y_0 = (double *)(Y + ldY * (r + 0));
                    for (c = cc; c < c_up; c++)
                    {
                        const double *x_0 = X + ldX * (c + 0) + r;
                        y_0[c + 0] = x_0[0];
                    }
                }
            }
        }
    }
}

void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const double *X, const ALPHA_INT ldX, double *__restrict Y, ALPHA_INT ldY)
{
    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float64x2_t v_row0, v_row1, v_row2, v_row3; //
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (ALPHA_INT r = 0; r < rowX; r++)
        {
            ALPHA_INT c = 0;
            //NPERCLF64 = 8 ; 512 / 128 = 4, thus need 4 register
            for (; c < colX - NPERCLF64 + 1; c += NPERCLF64)
            {
                const double *src = X + ldX * r + c;
                double *dst = Y + ldY * r + c;
#ifdef __aarch64__
                v_row0 = vld1q_f64(src + 0);
                v_row1 = vld1q_f64(src + 2);
                v_row2 = vld1q_f64(src + 4);
                v_row3 = vld1q_f64(src + 6);
                vst1q_f64(dst + 0, v_row0);
                vst1q_f64(dst + 2, v_row1);
                vst1q_f64(dst + 4, v_row2);
                vst1q_f64(dst + 6, v_row3);
#else
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
                dst[4] = src[4];
                dst[5] = src[5];
                dst[6] = src[6];
                dst[7] = src[7];
#endif
            }
            for (; c < colX - 1; c += 2)
            {
                const double *src = X + ldX * r + c;
                double *dst = Y + ldY * r + c;
#ifdef __aarch64__
                v_row0 = vld1q_f64(src + 0);
                vst1q_f64(dst + 0, v_row0);
#else
                dst[0] = src[0];
                dst[1] = src[1];
#endif
            }
            for (; c < colX; c++)
            {
                const double *src = X + ldX * r + c;
                double *dst = Y + ldY * r + c;
                dst[0] = src[0];
            }
        }
    }
}

//use double , may be hazardous

void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *__restrict Y, ALPHA_INT ldY)
{
    pack_r2c(rowX, colX, (double *)X, ldX, (double *)Y, ldY);
}

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *__restrict Y, ALPHA_INT ldY)
{
    pack_c2r(rowX, colX, (double *)X, ldX, (double *)Y, ldY);
}

void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex8 *X, const ALPHA_INT ldX, ALPHA_Complex8 *__restrict Y, ALPHA_INT ldY)
{
    pack_r2r(rowX, colX, (double *)X, ldX, (double *)Y, ldY);
}

void pack_c2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *__restrict Y, ALPHA_INT ldY)
{
    pack_matrix_col2row(rowX, colX, X, ldX, Y, ldY);
}
void pack_r2c(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *__restrict Y, ALPHA_INT ldY)
{
    pack_matrix_row2col(rowX, colX, X, ldX, Y, ldY);
}
void pack_r2r(const ALPHA_INT rowX, const ALPHA_INT colX, const ALPHA_Complex16 *X, const ALPHA_INT ldX, ALPHA_Complex16 *__restrict Y, ALPHA_INT ldY)
{

    ALPHA_INT num_threads = alpha_get_thread_num();
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
#ifdef __aarch64__
        float64x2_t v_row0, v_row1, v_row2, v_row3; //
#endif
#ifdef _OPENMP
#pragma omp for
#endif
        for (ALPHA_INT r = 0; r < rowX; r++)
        {
            ALPHA_INT c = 0;
            //NPERCLC16 = 4 ;
            for (; c < colX - NPERCLC16 + 1; c += NPERCLC16)
            {
                const double *src = (double *)(X + ldX * r + c);
                double *dst = (double *)(Y + ldY * r + c);
#ifdef __aarch64__
                v_row0 = vld1q_f64(src + 0);
                v_row1 = vld1q_f64(src + 2);
                v_row2 = vld1q_f64(src + 4);
                v_row3 = vld1q_f64(src + 6);
                vst1q_f64(dst + 0, v_row0);
                vst1q_f64(dst + 2, v_row1);
                vst1q_f64(dst + 4, v_row2);
                vst1q_f64(dst + 6, v_row3);
#else
                dst[0] = src[0];
                dst[1] = src[1];
                dst[2] = src[2];
                dst[3] = src[3];
#endif
            }
            for (; c < colX; c += 1)
            {
                const double *src = (double *)(X + ldX * r + c);
                double *dst = (double *)(Y + ldY * r + c);
#ifdef __aarch64__
                v_row0 = vld1q_f64(src + 0);
                vst1q_f64(dst + 0, v_row0);
#else
                dst[0] = src[0];
                dst[1] = src[1];
#endif
            }
        }
    }
}
