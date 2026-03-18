#pragma once


// #define _GNU_SOURCE
#include <vector>
#include <algorithm>
#include <numeric>
#include <memory.h>
#include <sched.h>
#include <immintrin.h>
#include "alphasparse/util.h"
#include "sellmv/sell_csigma_kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/bisearch.h"
#include "alphasparse/util/partition.h"
#include <type_traits>
#include <iostream> 
#include <omp.h>
#include "alphasparse/opt.h"
#include <cstring>

template <typename I, typename W>
alphasparseStatus_t gemv_sell_csigma(
    const double      alpha,
    const I           rows,
    const I           cols,
    const W*          row_ptr,
    const W*          col_data,
    const W*          block_max_nnz,
    const double*     val_data,
    const W           C,
    const W           sigma,
    const double*     x,
    const double      beta,
    double*           y,
    const W*          reorders)
{
    const I n_block_rows = rows / C;
    const I num_threads = alpha_get_thread_num();
    if(num_threads == 1){
        I tid = alpha_get_thread_id();
        I my_start = tid * n_block_rows;
        I my_end   = (tid + 1) * n_block_rows;
        for (I br = my_start; br < my_end; ++br)
        {
            I offset = row_ptr[br];
            I ml = block_max_nnz[br];

            for (I j = 0; j < C; j += 4)
            {
                __m256d acc = _mm256_setzero_pd();

                for (I i = 0; i < ml; ++i)
                {
                    I base = offset + i * C + j;
                    __m256d mat_val = _mm256_loadu_pd(val_data + base);

                    int idx0 = col_data[base + 0];
                    int idx1 = col_data[base + 1];
                    int idx2 = col_data[base + 2];
                    int idx3 = col_data[base + 3];

                    if (idx0 == -1) idx0 = 0;
                    if (idx1 == -1) idx1 = 0;
                    if (idx2 == -1) idx2 = 0;
                    if (idx3 == -1) idx3 = 0;

                    __m256d vec_val = _mm256_set_pd(
                        x[idx3], x[idx2], x[idx1], x[idx0]
                    );

                    acc = _mm256_fmadd_pd(mat_val, vec_val, acc);
                }

                double result[4];
                _mm256_storeu_pd(result, acc);

                I row_idx0 = br * C + j;
                I row_idx1 = row_idx0 + 1;
                I row_idx2 = row_idx0 + 2;
                I row_idx3 = row_idx0 + 3;

                if (row_idx0 < rows) y[reorders[row_idx0]] = alpha * result[0] + beta * y[reorders[row_idx0]];
                if (row_idx1 < rows) y[reorders[row_idx1]] = alpha * result[1] + beta * y[reorders[row_idx1]];
                if (row_idx2 < rows) y[reorders[row_idx2]] = alpha * result[2] + beta * y[reorders[row_idx2]];
                if (row_idx3 < rows) y[reorders[row_idx3]] = alpha * result[3] + beta * y[reorders[row_idx3]];
            }
        }
    }
    else{
                std::vector<long long> acc(n_block_rows);
        long long sum = 0;
        for (I br = 0; br < n_block_rows; ++br) {
            long long w = (long long)block_max_nnz[br] * (long long)C; // padded workload
            sum += w;
            acc[br] = sum;
        }
        std::vector<long long> partition(num_threads + 1);
        partition[0] = 0;
        if (n_block_rows > 0) {
            long long total = acc.back();
            long long ave = total / num_threads;
            partition[0] = 0;
            for (I i = 1; i < num_threads; ++i) {
                long long target = ave * i;
                I l = 0, r = (I)(n_block_rows - 1);
                while (r > l) {
                    I m = (l + r) >> 1;
                    if (acc[m] < target) l = m + 1;
                    else r = m;
                }
                partition[i] = l;
            }
            partition[num_threads] = n_block_rows;
        } else {
            for (I i = 0; i <= num_threads; ++i) partition[i] = 0;
        }
        #ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
        #endif
        {
            I tid = alpha_get_thread_id();
            I my_br_start = (I)partition[tid];
            I my_br_end   = (I)partition[tid + 1];

            for (I br = my_br_start; br < my_br_end; ++br)
            {
                I offset = row_ptr[br];
                I ml = block_max_nnz[br];

                for (I j = 0; j < C; j += 4)
                {
                    __m256d accv = _mm256_setzero_pd();

                    for (I i = 0; i < ml; ++i)
                    {
                        I base = offset + i * C + j;
                        __m256d mat_val = _mm256_loadu_pd(val_data + base);

                        int idx0 = col_data[base + 0];
                        int idx1 = col_data[base + 1];
                        int idx2 = col_data[base + 2];
                        int idx3 = col_data[base + 3];

                        if (idx0 == -1) idx0 = 0;
                        if (idx1 == -1) idx1 = 0;
                        if (idx2 == -1) idx2 = 0;
                        if (idx3 == -1) idx3 = 0;

                        __m256d vec_val = _mm256_set_pd(
                            x[idx3], x[idx2], x[idx1], x[idx0]
                        );

                        accv = _mm256_fmadd_pd(mat_val, vec_val, accv);
                    }

                    double result[4];
                    _mm256_storeu_pd(result, accv);

                    I row_idx0 = br * C + j;
                    I row_idx1 = row_idx0 + 1;
                    I row_idx2 = row_idx0 + 2;
                    I row_idx3 = row_idx0 + 3;

                    if (row_idx0 < rows) y[reorders[row_idx0]] = alpha * result[0] + beta * y[reorders[row_idx0]];
                    if (row_idx1 < rows) y[reorders[row_idx1]] = alpha * result[1] + beta * y[reorders[row_idx1]];
                    if (row_idx2 < rows) y[reorders[row_idx2]] = alpha * result[2] + beta * y[reorders[row_idx2]];
                    if (row_idx3 < rows) y[reorders[row_idx3]] = alpha * result[3] + beta * y[reorders[row_idx3]];
                }
            }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}


// float 
template <typename I, typename W>
alphasparseStatus_t gemv_sell_csigma(
    const float       alpha,
    const I           rows,
    const I           cols,
    const W*          row_ptr,
    const W*          col_data,
    const W*          block_max_nnz,
    const float*      val_data,
    const W           C,
    const W           sigma,
    const float*      x,
    const float       beta,
    float*            y,
    const W*          reorders)
{
    const I n_block_rows = rows / C;
    const I num_threads = alpha_get_thread_num();
    if(num_threads == 1){
        I tid = alpha_get_thread_id();
        I my_start = tid * n_block_rows;
        I my_end   = (tid + 1) * n_block_rows;

        for (I br = my_start; br < my_end; ++br)
        {
            I offset = row_ptr[br];
            I ml = block_max_nnz[br];

            for (I j = 0; j < C; j += 8)
            {
                __m256 acc = _mm256_setzero_ps();

                for (I i = 0; i < ml; ++i)
                {
                    I base = offset + i * C + j;
                    __m256 mat_val = _mm256_loadu_ps(val_data + base);

                    int idx[8];
                    for (int k = 0; k < 8; ++k) {
                        idx[k] = col_data[base + k];
                        if (idx[k] == -1) idx[k] = 0;
                    }

                    __m256 vec_val = _mm256_set_ps(
                        x[idx[7]], x[idx[6]], x[idx[5]], x[idx[4]],
                        x[idx[3]], x[idx[2]], x[idx[1]], x[idx[0]]
                    );

                    acc = _mm256_fmadd_ps(mat_val, vec_val, acc);
                }

                float result[8];
                _mm256_storeu_ps(result, acc);

                I row_idx[8];
                for (int k = 0; k < 8; ++k)
                    row_idx[k] = br * C + j + k;

                for (int k = 0; k < 8; ++k)
                {
                    if (row_idx[k] < rows)
                        y[reorders[row_idx[k]]] = alpha * result[k] + beta * y[reorders[row_idx[k]]];
                }
            }
        }
    }
    else{
        std::vector<long long> acc(n_block_rows);
        long long sum = 0;
        for (I br = 0; br < n_block_rows; ++br) {
            long long w = (long long)block_max_nnz[br] * (long long)C;
            sum += w;
            acc[br] = sum;
        }

        std::vector<long long> partition(num_threads + 1);
        partition[0] = 0;
        if (n_block_rows > 0) {
            long long total = acc.back();
            long long ave = total / num_threads;
            partition[0] = 0;
            for (I i = 1; i < num_threads; ++i) {
                long long target = ave * i;
                I l = 0, r = (I)(n_block_rows - 1);
                while (r > l) {
                    I m = (l + r) >> 1;
                    if (acc[m] < target) l = m + 1;
                    else r = m;
                }
                partition[i] = l;
            }
            partition[num_threads] = n_block_rows;
        } else {
            for (I i = 0; i <= num_threads; ++i) partition[i] = 0;
        }

    #ifdef _OPENMP
    #pragma omp parallel num_threads(num_threads)
    #endif
        {
            I tid = alpha_get_thread_id();
            I my_br_start = (I)partition[tid];
            I my_br_end   = (I)partition[tid + 1];

            for (I br = my_br_start; br < my_br_end; ++br)
            {
                I offset = row_ptr[br];
                I ml = block_max_nnz[br];

                for (I j = 0; j < C; j += 8)
                {
                    __m256 accv = _mm256_setzero_ps();

                    for (I i = 0; i < ml; ++i)
                    {
                        I base = offset + i * C + j;
                        __m256 mat_val = _mm256_loadu_ps(val_data + base);

                        int idx[8];
                        for (int k = 0; k < 8; ++k) {
                            idx[k] = col_data[base + k];
                            if (idx[k] == -1) idx[k] = 0;
                        }

                        __m256 vec_val = _mm256_set_ps(
                            x[idx[7]], x[idx[6]], x[idx[5]], x[idx[4]],
                            x[idx[3]], x[idx[2]], x[idx[1]], x[idx[0]]
                        );

                        accv = _mm256_fmadd_ps(mat_val, vec_val, accv);
                    }

                    float result[8];
                    _mm256_storeu_ps(result, accv);

                    I row_idx[8];
                    for (int k = 0; k < 8; ++k)
                        row_idx[k] = br * C + j + k;

                    for (int k = 0; k < 8; ++k)
                    {
                        if (row_idx[k] < rows)
                            y[reorders[row_idx[k]]] = alpha * result[k] + beta * y[reorders[row_idx[k]]];
                    }
                }
            }
        }
    }


    return ALPHA_SPARSE_STATUS_SUCCESS;
}


