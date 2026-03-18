#pragma once

#include "alphasparse.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_smblk_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    const U* x,
    U* y,
    void* externalBuffer
) {
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

// 以VECSIZE为一组进行规约
// VECSIZE可以为32、16、8、4、2、1
// 一个warp被划分为若干个VECSIZE大小的vector
template<unsigned int VECSIZE, typename U>
__device__ __forceinline__ static U
vec_reduce_sum(
    unsigned int vec_id,    // 在同一个warp内，当前线程属于第vec_id的vector
    U num
) {
    unsigned int MASK = 0xFFFFFFFF;
    if (VECSIZE < 32) {
        MASK = (1u << VECSIZE) - 1;
        MASK = MASK << (vec_id * VECSIZE);
    }
    
    for (int offset = VECSIZE >> 1; offset > 0; offset >>= 1) {
        num += __shfl_xor_sync(MASK, num, offset,VECSIZE);
    }
    return num;
}

// 1 warp 处理 1 row
template <unsigned int BLOCKSIZE, unsigned int VECSIZE, typename T>
__global__ static void
spsv_csr_n_lo_smblk_wpr_solve_kernel_volatile(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    volatile T* __restrict__ get_value
) {
    T lid = threadIdx.x & (VECSIZE - 1);
    T vec_id = threadIdx.x / VECSIZE;           // 在当前block中是第几个vec
    T sub_vec_id = vec_id % (32 / VECSIZE);    // 在32个线程为一组的warp里，是第几个vec
    T stride = blockDim.x / VECSIZE * gridDim.x;
    for (T idx = blockIdx.x * (blockDim.x / VECSIZE) + vec_id; idx < m; idx += stride) {
        T row = __ldg(&row_map[idx]);
        T row_begin = __ldg(&csr_row_ptr[row]);
        T row_end = __ldg(&csr_row_ptr[row + 1]);
        double local_sum = {};
        if (lid == 0) {
            local_sum = alpha * __ldg(&x[row]);
        } 
        T j = row_begin + lid;
        T local_col = m;
        double local_val = {};
        if (j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
            local_val = __ldg(&csr_val[j]);
        } 
        while (j < row_end && local_col < row) {
            int t = get_value[local_col];
            j += t * VECSIZE;
            if (t) {
                local_sum -= local_val * y[local_col];
            }
            if (t && j < row_end) { 
                local_col = __ldg(&csr_col_idx[j]);
                local_val = __ldg(&csr_val[j]);
            }
        }
        local_sum = vec_reduce_sum<VECSIZE>(sub_vec_id, local_sum);
        if (local_col == row) {
            y[row] = local_sum / local_val;
            __threadfence();
            get_value[row] = 1;
        }
        __syncthreads();
    }
    return;
}

// 1 thread 处理 1 row
template <typename T>
__global__ static void
spsv_csr_n_lo_smblk_tpr_solve_kernel_volatile(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    volatile T* __restrict__ get_value
) {
    const int stride = blockDim.x * gridDim.x;
    for (int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < m; tid += stride) {
        T row_id = __ldg(&row_map[tid]);
        // calculate tmp_sum
        double tmp_sum = alpha * __ldg(&x[row_id]);
        T row_end = __ldg(&csr_row_ptr[row_id + 1]);
        T ptr = __ldg(&csr_row_ptr[row_id]);
        T col_id = __ldg(&csr_col_idx[ptr]);
        while (ptr < row_end) {
            if (col_id == row_id) {
                // write y
                y[row_id] = tmp_sum / __ldg(&csr_val[ptr]);
                __threadfence();
                get_value[row_id] = 1;
                break;
            } else if (get_value[col_id] == 1) {
                tmp_sum = fma(-__ldg(&csr_val[ptr]), y[col_id], tmp_sum);
                ptr++;
                col_id = __ldg(&csr_col_idx[ptr]);
            }
        }
        __syncthreads();
    }
    return;
}


#define KERNEL_DISPATCH(BLOCKSIZE, VECSIZE) {                               \
    if (VECSIZE > 1) {                                                      \
        spsv_csr_n_lo_smblk_wpr_solve_kernel_volatile<BLOCKSIZE, VECSIZE>  \
        <<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(              \
            csr_row_ptr,                                                    \
            csr_col_idx,                                                    \
            csr_val,                                                        \
            m,                                                              \
            nnz,                                                            \
            alpha,                                                          \
            x,                                                              \
            y,                                                              \
            row_map,                                                        \
            get_value                                                       \
        );                                                                  \
    } else {                                                                \
        spsv_csr_n_lo_smblk_tpr_solve_kernel_volatile             \
        <<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(              \
            csr_row_ptr,                                                    \
            csr_col_idx,                                                    \
            csr_val,                                                        \
            m,                                                              \
            nnz,                                                            \
            alpha,                                                          \
            x,                                                              \
            y,                                                              \
            row_map,                                                        \
            get_value                                                       \
        );                                                                  \
    }                                                                       \
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_smblk_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    const U* x,
    U* y,
    void *externalBuffer
) {
    // const unsigned int sm_num = 80;
    const unsigned int sm_num = 160;
    // const unsigned int sm_num = 240;
    // const unsigned int sm_num = 320;
    // const unsigned int sm_num = 400;
    // const unsigned int sm_num = 640;
    // const unsigned int sm_num = 1280;
    // const unsigned int sm_num = 2560;
    // const unsigned int BLOCKSIZE = 64;
    // const unsigned int BLOCKSIZE = 128;
    const unsigned int BLOCKSIZE = 256;
    // const unsigned int BLOCKSIZE = 512;
    // const unsigned int BLOCKSIZE = 1024;
    
    // 1 warp
    dim3 threadPerBlock(BLOCKSIZE);
    dim3 blockPerGrid(sm_num);

    T *get_value = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(get_value, 0, m * sizeof(T));
    
    const unsigned int VECSIZE = 32;

    KERNEL_DISPATCH(BLOCKSIZE, VECSIZE);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}