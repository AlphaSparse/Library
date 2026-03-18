#pragma once

#include "alphasparse.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>


// done_array denotes each row's level
template<unsigned int WARP_SIZE, typename T>
static __global__ void
spsv_csr_n_lo_smblk_rearrange_analysis_getlevel_kernel(
    const T m,
    const T nnz,
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    volatile T* __restrict__ done_array,
    T* __restrict__ row_map,
    T* __restrict__ row_length
) {
    // 1 warp 1 row
    T global_id = blockIdx.x * blockDim.x + threadIdx.x;
    T row_id = global_id >> 5;
    T lane_id = global_id & 0x1F;
    // T row_id = global_id / WARP_SIZE;
    // T lane_id = global_id % WARP_SIZE;
    if (row_id >= m) {
        return;
    }
    if (lane_id == 0) {
        row_map[row_id] = row_id;
    }
    T local_max = 0;
    T local_nnz = 0;
    T row_begin = csr_row_ptr[row_id];
    T row_end = csr_row_ptr[row_id + 1];
    T ptr = row_begin + lane_id;
    while (ptr < row_end) {
        T col_id = csr_col_idx[ptr];
        if (col_id >= row_id) {
            break;
        }
        local_nnz++;
        T local_done = done_array[col_id];
        while (local_done == 0) {
            local_done = done_array[col_id];
            // if (row_id < 10000)
            // printf("row_id: %d, col_id: %d\n", row_id, col_id);
        }
        local_max = max(local_max, local_done);
        ptr += WARP_SIZE;
    }
    local_max = warp_reduce_max<WARP_SIZE>(local_max);
    warp_reduce_sum<WARP_SIZE>(&local_nnz);
    if (lane_id == WARP_SIZE - 1) {
        done_array[row_id] = local_max + 1;
        row_length[row_id] = local_nnz + 1;
    }
    return;
}

template<unsigned int WARP_SIZE, typename T, typename U>
static __global__ void
spsv_csr_n_lo_smblk_rearrange_analysis_updatecsr_kernel(
    const T m,
    const T nnz,
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const U* __restrict__ csr_val,
    const T* __restrict__ row_map,
    const T* __restrict__ rcsr_row_ptr,
    T* __restrict__ rcsr_col_idx,
    U* __restrict__ rcsr_val
) {
    // 1 thread 1 row
    T rrow_id = blockIdx.x * blockDim.x + threadIdx.x; // 相对于rcsr_row_ptr
    if (rrow_id >= m) {
        return;
    }
    T row_id = row_map[rrow_id];    // 相对于csr_row_ptr
    T rrow_begin = rcsr_row_ptr[rrow_id];
    T rrow_end = rcsr_row_ptr[rrow_id + 1];
    T row_begin = csr_row_ptr[row_id];
    T rptr = rrow_begin;
    T ptr = row_begin;
    while (rptr < rrow_end) {
        T col_id = csr_col_idx[ptr];
        U val = csr_val[ptr];
        rcsr_col_idx[rptr] = col_id;
        rcsr_val[rptr] = val;
        rptr++;
        ptr++;
    }
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_smblk_rearrange_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    T* rcsr_row_ptr,
    T* rcsr_col_idx,
    U* rcsr_val,
    const U* x,
    // U* x,
    U* y,
    void* externalBuffer
) {
    const unsigned int BLOCKSIZE = 512;
    const unsigned int WARP_SIZE = 32;
    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid = dim3((m * WARP_SIZE - 1) / BLOCKSIZE + 1);
    T *done_array = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(done_array, 0, m * sizeof(T));
    T* row_length;
    cudaMalloc((void **)&row_length, (m + 1) * sizeof(T));
    cudaMemset(row_length, 0, (m + 1) * sizeof(T));
    spsv_csr_n_lo_smblk_rearrange_analysis_getlevel_kernel<WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        m,
        nnz,
        csr_row_ptr,
        csr_col_idx,
        done_array,
        row_map,
        row_length
    );
    // sort done_array and row_map
    T* done_array_o;
    cudaMalloc((void **)&done_array_o, m * sizeof(T));
    get_row_map_sorted(m, done_array, done_array_o, row_map, row_map);
    get_row_map_sorted(m, done_array, done_array_o, row_length, row_length);
    // get_row_map_sorted(m, done_array, done_array_o, x, x);
    cudaFree(done_array_o);
    // rearrange csr_row_ptr, csr_col_idx
    // T* csr_row_ptr_o;
    // cudaMalloc((void **)&csr_row_ptr_o, (m + 1) * sizeof(T));
    // T* csr_col_idx_o;
    // cudaMalloc((void **)&csr_col_idx_o, nnz * sizeof(T)); // 暂用总NNZ数设定重排后到col_idx数组长度
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        row_length,
        rcsr_row_ptr,
        m + 1
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_storage_bytes,
        row_length,
        rcsr_row_ptr,
        m + 1
    );
    blockPerGrid = dim3((m - 1) / BLOCKSIZE + 1);
    spsv_csr_n_lo_smblk_rearrange_analysis_updatecsr_kernel<WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        m,
        nnz,
        csr_row_ptr,
        csr_col_idx,
        csr_val,
        row_map,
        rcsr_row_ptr,
        rcsr_col_idx,
        rcsr_val
    );
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

// 1 warp 处理 1 row
template <unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_smblk_rearrange_wpr_solve_kernel_volatile(
    const T* __restrict__ rcsr_row_ptr,
    const T* __restrict__ rcsr_col_idx,
    const double* __restrict__ rcsr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    volatile bool* __restrict__ get_value
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T stride = blockDim.x / WARP_SIZE * gridDim.x;
    for (T idx = blockIdx.x * (blockDim.x / WARP_SIZE) + wid; idx < m; idx += stride) {
        T rrow = idx;
        T row = __ldg(&row_map[idx]);
        T row_begin = __ldg(&rcsr_row_ptr[rrow]);
        T row_end = __ldg(&rcsr_row_ptr[rrow + 1]);
        double local_sum = {};
        if (lid == 0) {
            local_sum = alpha * __ldg(&x[row]);
        }
        T j = row_begin + lid;
        T local_col = m;
        double local_val = {};
        if (j < row_end) {
            local_col = __ldg(&rcsr_col_idx[j]);
            local_val = __ldg(&rcsr_val[j]);
        } 
        while (j < row_end && local_col < row) {
            int t = get_value[local_col];
            t = __all_sync(__activemask(), t) ? 1 : 0;
            j += t * WARP_SIZE;
            if (t) {
                local_sum -= local_val * y[local_col];
            }
            if (t && j < row_end) { 
                local_col = __ldg(&rcsr_col_idx[j]);
                local_val = __ldg(&rcsr_val[j]);
            }
        }
        warp_reduce_sum<WARP_SIZE>(&local_sum);
        if (local_col == row) {
            y[row] = local_sum / local_val;
            __threadfence();
            get_value[row] = 1;
        }
        __syncthreads();
    }
    return;
}


// bitmap存储get_value
template <unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_smblk_rearrange_wpr_solve_kernel_nonvolatile(
    const T* __restrict__ rcsr_row_ptr,
    const T* __restrict__ rcsr_col_idx,
    const double* __restrict__ rcsr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    uint32_t* __restrict__ get_value
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T stride = blockDim.x / WARP_SIZE * gridDim.x;
    for (T idx = blockIdx.x * (blockDim.x / WARP_SIZE) + wid; idx < m; idx += stride) {
        T rrow = idx;
        T row = __ldg(&row_map[idx]);
        T row_begin = __ldg(&rcsr_row_ptr[rrow]);
        T row_end = __ldg(&rcsr_row_ptr[rrow + 1]);
        double local_sum = {};
        if (lid == 0) {
            local_sum = alpha * __ldg(&x[row]);
        }
        T j = row_begin + lid;
        T local_col = m;
        double local_val = {};
        if (j < row_end) {
            local_col = __ldg(&rcsr_col_idx[j]);
            local_val = __ldg(&rcsr_val[j]);
        } 
        while (j < row_end && local_col < row) {
            // int t = atomicOr(&get_value[local_col], 0);
            uint32_t bits = atomicOr(&get_value[local_col >> 5], 0);
            // int t = (bits & (1 << (local_col % 32)));
            int t = (bits & (1 << (local_col & 31)));
            t = __all_sync(__activemask(), t) ? 1 : 0;
            j += t * WARP_SIZE;
            if (t) {
                // __threadfence();
                local_sum -= local_val * y[local_col];
            }
            if (t && j < row_end) { 
                local_col = __ldg(&rcsr_col_idx[j]);
                local_val = __ldg(&rcsr_val[j]);
            }
        }
        warp_reduce_sum<WARP_SIZE>(&local_sum);
        if (local_col == row) {
            y[row] = local_sum / local_val;
            __threadfence();
            // atomicOr(&get_value[row], 1);
            // atomicOr(&get_value[row >> 5], (1 << (row % 32)));
            atomicOr(&get_value[row >> 5], (1 << (row & 31)));
        }
        __syncthreads();
    }
    return;
}


template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_smblk_rearrange_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* rcsr_val,
    const T* rcsr_row_ptr,
    const T* rcsr_col_idx,
    T* row_map,
    const U* x,
    U* y,
    void *externalBuffer
) {
    const unsigned int sm_num = 80;
    const unsigned int BLOCKSIZE = 64;
    // 1 warp
    dim3 threadPerBlock(BLOCKSIZE);
    dim3 blockPerGrid(sm_num);

    uint32_t *get_value = reinterpret_cast<uint32_t*>(externalBuffer);
    cudaMemset(get_value, 0, ((m >> 5) + 1) * sizeof(uint32_t));

    // printf("threadPerBlock: %d\n", threadPerBlock.x);
    spsv_csr_n_lo_smblk_rearrange_wpr_solve_kernel_nonvolatile<BLOCKSIZE, 32><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
    // spsv_csr_n_lo_smblk_rearrange_wpr_solve_kernel_volatile<BLOCKSIZE, 32><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
    // spsv_csr_n_lo_smblk_rearrange_tpr_solve_kernel_nonvolatile<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        rcsr_row_ptr,
        rcsr_col_idx,
        rcsr_val,
        m,
        nnz,
        alpha,
        x,
        y,
        row_map,
        get_value
    );

    return ALPHA_SPARSE_STATUS_SUCCESS;
}