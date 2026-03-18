#pragma once

#include "alphasparse.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// alg_num: 9

template<unsigned int WARP_SIZE, typename U>
__device__ __forceinline__ static U
warp_reduce_max(
    const U num
) {
    U tmp_num = num;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        tmp_num = max(tmp_num, __shfl_xor_sync(0xFFFFFFFF, tmp_num, offset, WARP_SIZE));
    }
    return tmp_num;
}

template<unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void 
spsv_csr_n_lo_roc_analysis_kernel(
    const T m,
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    volatile T* __restrict__ done_array,
    T* __restrict__ row_map,
    const T row_offset
){
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T first_row = row_offset + blockIdx.x * (blockDim.x / WARP_SIZE);
    T row = first_row + wid;
    volatile __shared__ T local_done_array[BLOCKSIZE / WARP_SIZE];
    if (lid == 0) {
        local_done_array[wid] = 0;
    }
    __syncthreads();
    if (row >= m) {
        return;
    }
    if (lid == 0) {
        row_map[row] = row;
    }
    T local_max = 0;
    T row_begin = csr_row_ptr[row];
    T row_end = csr_row_ptr[row + 1];
    T j = row_begin + lid;
    T local_col = __ldg(&csr_col_idx[j]);
    while (j < row_end && local_col < first_row) {
        T local_done = done_array[local_col];
        local_max = max(local_done, local_max);
        j += (local_done != 0) * WARP_SIZE;
        if (local_done != 0 && j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
        }
    }
    while (j < row_end && local_col < row) {
        T local_idx = local_col - first_row;
        T local_done = local_done_array[local_idx];
        j += (local_done != 0) * WARP_SIZE;
        local_max = max(local_done, local_max);
    }
    local_max = warp_reduce_max<WARP_SIZE>(local_max);
    if (lid == WARP_SIZE - 1) {
        local_done_array[wid] = local_max + 1;
        done_array[row] = local_max + 1;
    }
    return;
}

template<typename T, typename U>
void
get_row_map_sorted(
    const T m,
    T* done_array_i,
    T* done_array_o,
    U* row_map_i,
    U* row_map_o
) {
    thrust::device_vector<T> keys(done_array_i, done_array_i + m);
    thrust::device_vector<U> values(row_map_i, row_map_i + m);
    thrust::sort_by_key(keys.begin(), keys.end(), values.begin());
    cudaMemcpy(done_array_o, thrust::raw_pointer_cast(keys.data()), m * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpy(row_map_o, thrust::raw_pointer_cast(values.data()), m * sizeof(U), cudaMemcpyDeviceToDevice);
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_roc_analysis(
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
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 32;
    const int maxThreadPerGrid = 1 << 30;
    const int rowPerGrid = maxThreadPerGrid / WARP_SIZE;

    const dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid;

    T *done_array = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(done_array, 0, m * sizeof(T));
    for (T row_offset = 0, row_remain = m; row_offset < m; row_offset += rowPerGrid, row_remain -= rowPerGrid) {
        blockPerGrid = dim3((min(row_remain, rowPerGrid) - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
        // printf("blockPerGrid = %d, row_offset = %d\n", blockPerGrid.x, row_offset);
        spsv_csr_n_lo_roc_analysis_kernel<BLOCKSIZE, WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
            m,
            csr_row_ptr,
            csr_col_idx,
            done_array,
            row_map,
            row_offset
        );
        // CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // cudaDeviceSynchronize();
    get_row_map_sorted(m, done_array, done_array, row_map, row_map);

    // T *h_row_map = (T*)malloc(m * sizeof(T));
    // T *h_done_array = (T*)malloc(m * sizeof(T));
    // cudaMemcpy(h_row_map, row_map, m * sizeof(T), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_done_array, done_array, m * sizeof(T), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < m; i++) {
    //     printf("done_array[%d] = %d, row_map[%d] = %d\n", i, h_done_array[i], i, h_row_map[i]);
    // }
    // exit(0);
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

// lane_id = WARP_SIZE - 1 得到累加值
template<unsigned int WARP_SIZE, typename U>
__device__ __forceinline__ static U
warp_reduce_sum(
    U *num
) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        *num += __shfl_xor_sync(0xFFFFFFFF, *num, offset, WARP_SIZE);
    }
    return *num;
}

template<unsigned int WARP_SIZE>
__device__ __forceinline__ static cuDoubleComplex
warp_reduce_sum(
    cuDoubleComplex *num
) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        (*num).x += __shfl_xor_sync(0xFFFFFFFF, (*num).x, offset, WARP_SIZE);
        (*num).y += __shfl_xor_sync(0xFFFFFFFF, (*num).y, offset, WARP_SIZE);
    }
    return *num;
}

template<unsigned int WARP_SIZE>
__device__ __forceinline__ static cuFloatComplex
warp_reduce_sum(
    cuFloatComplex *num
) {
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        (*num).x += __shfl_up_sync(0xFFFFFFFF, (*num).x, offset, WARP_SIZE);
        (*num).y += __shfl_up_sync(0xFFFFFFFF, (*num).y, offset, WARP_SIZE);
    }
    return *num;
}

template<typename U>
__device__ __forceinline__ static U
my_fma(
    U a,
    U b,
    U c
) {
    return fma(a, b, c);
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    volatile T* __restrict__ done_array,
    const T row_offset
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T idx = row_offset + blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    volatile __shared__ double diag[BLOCKSIZE / WARP_SIZE];
    if (idx >= m) {
        return;
    }
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
        int t = done_array[local_col];
        // while (t == 0) {
        //     t = done_array[local_col];
        // }
        t = __all_sync(__activemask(), t) ? 1 : 0;
        // __shfl_xor_sync(-1, t, )
        j += t * WARP_SIZE;
        if (t) {
            // local_sum = my_fma(-local_val, y[local_col], local_sum);
            local_sum += -local_val * y[local_col];
        }
        if (t && j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
            local_val = __ldg(&csr_val[j]);
        }
    }
    if (local_col == row) {
        diag[wid] = double(1) / local_val;
    }
    warp_reduce_sum<WARP_SIZE>(&local_sum);
    if (lid == WARP_SIZE - 1) {
        y[row] = local_sum * diag[wid];
        __threadfence();
        done_array[row] = 1;
    }
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const float* __restrict__ csr_val,
    const T m,
    const T nnz,
    const float alpha,
    const float* __restrict__ x,
    volatile float* __restrict__ y,
    const T* __restrict__ row_map,
    volatile T* __restrict__ done_array,
    const T row_offset
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T idx = row_offset + blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    volatile __shared__ float diag[BLOCKSIZE / WARP_SIZE];
    if (idx >= m) {
        return;
    }
    T row = __ldg(&row_map[idx]);
    T row_begin = __ldg(&csr_row_ptr[row]);
    T row_end = __ldg(&csr_row_ptr[row + 1]);
    float local_sum = {};
    if (lid == 0) {
        local_sum = alpha * __ldg(&x[row]);
    }
    T j = row_begin + lid;
    T local_col = m;
    float local_val = {};
    if (j < row_end) {
        local_col = __ldg(&csr_col_idx[j]);
        local_val = __ldg(&csr_val[j]);
    } 
    while (j < row_end && local_col < row) {
        int t = (done_array[local_col] != 0);
        j += t * WARP_SIZE;
        if (t) {
            local_sum = my_fma(-local_val, y[local_col], local_sum);
        }
        if (t && j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
            local_val = __ldg(&csr_val[j]);
        }
    }
    if (local_col == row) {
        diag[wid] = float(1) / local_val;
    }
    warp_reduce_sum<WARP_SIZE>(&local_sum);
    if (lid == WARP_SIZE - 1) {
        y[row] = local_sum * diag[wid];
        __threadfence();
        done_array[row] = 1;
    }
    return;
}


template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const cuDoubleComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* __restrict__ x,
    cuDoubleComplex* __restrict__ y,
    const T* __restrict__ row_map,
    volatile T* __restrict__ done_array,
    const T row_offset
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T idx = row_offset + blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    volatile __shared__ cuDoubleComplex diag[BLOCKSIZE / WARP_SIZE];
    if (idx >= m) {
        return;
    }
    T row = __ldg(&row_map[idx]);
    T row_begin = __ldg(&csr_row_ptr[row]);
    T row_end = __ldg(&csr_row_ptr[row + 1]);
    cuDoubleComplex local_sum = {};
    if (lid == 0) {
        local_sum = alpha * __ldg(&x[row]);
    }
    T j = row_begin + lid;
    T local_col = m;
    cuDoubleComplex local_val = {};
    if (j < row_end) {
        local_col = __ldg(&csr_col_idx[j]);
        local_val = __ldg(&csr_val[j]);
    } 
    while (j < row_end && local_col < row) {
        int t = (done_array[local_col] != 0);
        j += t * WARP_SIZE;
        if (t) {
            __threadfence();
            // local_sum = my_fma(-local_val, y[local_col], local_sum);
            local_sum = local_sum - local_val * y[local_col];
        }
        if (t && j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
            local_val = __ldg(&csr_val[j]);
        }
    }
    if (local_col == row) {
        // diag[wid] = cuDoubleComplex{1, 1} / local_val;
        diag[wid].x = local_val.x;
        diag[wid].y = local_val.y;
    }
    warp_reduce_sum<WARP_SIZE>(&local_sum);
    if (lid == WARP_SIZE - 1) {
        cuDoubleComplex diagi;
        diagi.x = diag[wid].x;
        diagi.y = diag[wid].y;
        y[row] = local_sum / diagi;
        __threadfence();
        done_array[row] = 1;
    }
    return;
}

template<unsigned int BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_roc_solve_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const cuFloatComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const cuFloatComplex alpha,
    const cuFloatComplex* __restrict__ x,
    cuFloatComplex* __restrict__ y,
    const T* __restrict__ row_map,
    volatile T* __restrict__ done_array,
    const T row_offset
) {
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T idx = row_offset + blockIdx.x * (blockDim.x / WARP_SIZE) + wid;
    volatile __shared__ cuFloatComplex diag[BLOCKSIZE / WARP_SIZE];
    if (idx >= m) {
        return;
    }
    T row = __ldg(&row_map[idx]);
    T row_begin = __ldg(&csr_row_ptr[row]);
    T row_end = __ldg(&csr_row_ptr[row + 1]);
    cuFloatComplex local_sum = {};
    if (lid == 0) {
        local_sum = alpha * __ldg(&x[row]);
    }
    T j = row_begin + lid;
    T local_col = m;
    cuFloatComplex local_val = {};
    if (j < row_end) {
        local_col = __ldg(&csr_col_idx[j]);
        local_val = __ldg(&csr_val[j]);
    } 
    while (j < row_end && local_col < row) {
        int t = (done_array[local_col] != 0);
        j += t * WARP_SIZE;
        if (t) {
            __threadfence();
            // local_sum = my_fma(-local_val, y[local_col], local_sum);
            local_sum += -local_val * y[local_col];
        }
        if (t && j < row_end) {
            local_col = __ldg(&csr_col_idx[j]);
            local_val = __ldg(&csr_val[j]);
        }
    }
    if (local_col == row) {
        diag[wid].x = local_val.x;
        diag[wid].y = local_val.y;
    }
    warp_reduce_sum<WARP_SIZE>(&local_sum);
    if (lid == WARP_SIZE - 1) {
        cuFloatComplex diagi;
        diagi.x = diag[wid].x;
        diagi.y = diag[wid].y;
        y[row] = local_sum / diagi;
        __threadfence();
        done_array[row] = 1;
    }
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_roc_solve(
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
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 32;
    const int maxThreadPerGrid = 1 << 30;
    const int rowPerGrid = maxThreadPerGrid / WARP_SIZE;

    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid;

    T *done_array = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(done_array, 0, m * sizeof(T));

    for (T row_offset = 0, row_remain = m; row_offset < m; row_offset += rowPerGrid, row_remain -= rowPerGrid) {
        blockPerGrid = dim3((min(row_remain, rowPerGrid) - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
        spsv_csr_n_lo_roc_solve_kernel<BLOCKSIZE, WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
            csr_row_ptr,
            csr_col_idx,
            csr_val,
            m,
            nnz,
            alpha,
            x,
            y,
            row_map,
            done_array,
            row_offset
        );
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}