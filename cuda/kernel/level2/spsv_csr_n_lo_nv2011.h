#pragma once

#include "alphasparse.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// level-scheduling nvidia Naumov 2011
// single block multiple levels OR multiple blocks single level 
// 1 thread 1 row

// ./build/src/test/spsv_csr_r_f64_test_metrics --data-file=../matrix_test/2cubes_sphere.mtx --transA=N --fillA=L --diagA=N --iter=1 --warmup=0 --alg_num=12 --check --metrics



template<typename T>
__global__ static void 
get_level_ptr(
    const T* __restrict__ done_array,
    const T m,
    T* __restrict__ level_ptr,
    T* level_size
) {
    const T tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= m) {
        return;
    }
    T cur_degree = done_array[tid];
    if (tid == 0 || cur_degree != done_array[tid - 1]) {
        level_ptr[cur_degree - 1] = tid;
    }
    if (tid == m - 1) {
        level_ptr[cur_degree] = m;
        *level_size = cur_degree + 1;
    }
    return;
}

template<unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void 
spsv_csr_n_lo_nv2011_analysis_kernel(
    const T m,
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    volatile T* __restrict__ done_array,
    T* __restrict__ row_map
){
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T first_row = blockIdx.x * (blockDim.x / WARP_SIZE);
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
    T local_col = csr_col_idx[j];
    // T local_col = __builtin_nontemporal_load(&csr_col_idx[j]);
    while (j < row_end && local_col < first_row) {
        T local_done = done_array[local_col];
        local_max = max(local_done, local_max);
        j += (local_done != 0) * WARP_SIZE;
        if (local_done != 0 && j < row_end) {
            local_col = csr_col_idx[j];
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

template<typename T>
__host__ static void 
get_chain_ptr(
    const T* h_level_ptr,
    T h_level_size,
    T* h_chain_ptr,
    T* h_chain_size
) {
    // h_chain_ptr = (T *)malloc(h_level_size * sizeof(T));
    h_chain_ptr[0] = 0;
    int ptr = 1;
    const T threshold = 1024;
    // const T threshold = 64;
    int i = 1;
    while (i < h_level_size) {
        if (h_level_ptr[i] - h_level_ptr[i - 1] > threshold) {
            if (h_chain_ptr[ptr - 1] != i - 1) {
                h_chain_ptr[ptr++] = i - 1;
            }
            h_chain_ptr[ptr++] = i;
        }
        i++;
    }
    if (h_chain_ptr[ptr - 1] != h_level_size - 1) {
        h_chain_ptr[ptr++] = h_level_size - 1;
    }
    *h_chain_size = ptr;
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_nv2011_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    T* d_level_ptr,
    T* h_level_size,
    T* h_chain_ptr,
    T* h_chain_size,
    const U* x,
    U* y,
    void* externalBuffer
) {
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 32;  // 设置为32 - 运行时死锁？
    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid = dim3((m - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
    T *done_array = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(done_array, 0, m * sizeof(T));
    spsv_csr_n_lo_nv2011_analysis_kernel<BLOCKSIZE, WARP_SIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        m,
        csr_row_ptr,
        csr_col_idx,
        done_array,
        row_map
    );
    // T *done_array_o;
    get_row_map_sorted(m, done_array, done_array, row_map, row_map);

    T *d_level_size;
    cudaMalloc((void **)&d_level_size, sizeof(T));
    blockPerGrid = dim3((m - 1) / threadPerBlock.x + 1);
    get_level_ptr<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>( 
        done_array,
        m,
        d_level_ptr,
        d_level_size
    );
    cudaMemcpy(h_level_size, d_level_size, sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_level_size);
    // printf("%d,%d,%d,%d,", nnz, m, *h_level_size, m/(*h_level_size));
    // printf("\n");
    T *h_level_ptr = (T *)malloc(*h_level_size * sizeof(T));
    cudaMemcpy(h_level_ptr, d_level_ptr, *h_level_size * sizeof(T), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; i++) {
    //     printf("level_ptr[%d] = %d\n", i, h_level_ptr[i]);
    // }
    cudaDeviceSynchronize();
    get_chain_ptr(
        h_level_ptr,
        *h_level_size,
        h_chain_ptr,
        h_chain_size
    );
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_sbml_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T chain_begin,
    const T chain_end
) {
    for (int level_id = chain_begin; level_id < chain_end; level_id++) {
        const T level_begin = level_ptr[level_id];
        const T level_end = level_ptr[level_id + 1];
        for (int idx = level_begin + threadIdx.x; idx < level_end; idx += blockDim.x) {
            const T row_id = __ldg(&row_map[idx]);
            const T row_begin = __ldg(&csr_row_ptr[row_id]);
            const T row_end = __ldg(&csr_row_ptr[row_id + 1]);
            double tmp_sum = alpha * __ldg(&x[row_id]);
            for (int val_id = row_begin; val_id < row_end; val_id++) {
                T col_id = __ldg(&csr_col_idx[val_id]);
                if (col_id > row_id) {
                    break;
                }
                if (col_id == row_id) {
                    tmp_sum /= __ldg(&csr_val[val_id]);
                    break;
                }
                tmp_sum -= __ldg(&csr_val[val_id]) * y[col_id];
            }
            y[row_id] = tmp_sum;
        }
        __syncthreads();
    }
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_sbml_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const float* __restrict__ csr_val,
    const T m,
    const T nnz,
    const float alpha,
    const float* __restrict__ x,
    volatile float* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T chain_begin,
    const T chain_end
) {
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_sbml_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const cuDoubleComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* __restrict__ x,
    volatile cuDoubleComplex* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T chain_begin,
    const T chain_end
) {
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_sbml_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const cuFloatComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const cuFloatComplex alpha,
    const cuFloatComplex* __restrict__ x,
    volatile cuFloatComplex* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T chain_begin,
    const T chain_end
) {
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_mbsl_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const double* __restrict__ csr_val,
    const T m,
    const T nnz,
    const double alpha,
    const double* __restrict__ x,
    volatile double* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_id
) {
    const T tid = blockIdx.x * blockDim.x + threadIdx.x;
    const T level_begin = level_ptr[level_id];
    const T level_end = level_ptr[level_id + 1];
    for (int idx = level_begin + tid; idx < level_end; idx += gridDim.x) {
        const T row_id = __ldg(&row_map[idx]);
        const T row_begin = __ldg(&csr_row_ptr[row_id]);
        const T row_end = __ldg(&csr_row_ptr[row_id + 1]);
        double tmp_sum = alpha * __ldg(&x[row_id]);
        for (int val_id = row_begin; val_id < row_end; val_id++) {
            T col_id = __ldg(&csr_col_idx[val_id]);
            if (col_id > row_id) {
                break;
            }
            if (col_id == row_id) {
                tmp_sum /= __ldg(&csr_val[val_id]);
                break;
            }
            tmp_sum -= __ldg(&csr_val[val_id]) * y[col_id];
        }
        y[row_id] = tmp_sum;
    }
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_mbsl_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const float* __restrict__ csr_val,
    const T m,
    const T nnz,
    const float alpha,
    const float* __restrict__ x,
    volatile float* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_id
) {
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_mbsl_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const cuDoubleComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* __restrict__ x,
    volatile cuDoubleComplex* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_id
) {
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_nv2011_solve_mbsl_kernel(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const cuFloatComplex* __restrict__ csr_val,
    const T m,
    const T nnz,
    const cuFloatComplex alpha,
    const cuFloatComplex* __restrict__ x,
    volatile cuFloatComplex* __restrict__ y,
    const T* __restrict__ row_map,
    const T* __restrict__ level_ptr,
    const T level_id
) {
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_nv2011_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* row_map,
    const T* d_level_ptr,
    const T h_level_size,
    const T* h_chain_ptr,
    const T h_chain_size,
    const U* x,
    U* y,
    void *externalBuffer
) {
    const unsigned int BLOCKSIZE = 1024;
    const unsigned int WARP_SIZE = 32;

    dim3 threadPerBlock = dim3(BLOCKSIZE);
    dim3 blockPerGrid;

    // T *done_array = reinterpret_cast<T*>(externalBuffer);
    // cudaMemset(done_array, 0, m * sizeof(T));


    // for (int i = 0; i < h_chain_size; i++) {
    //     printf("chain_ptr[%d] = %d\n", i, h_chain_ptr[i]);
    // }
    // cudaDeviceSynchronize();

    for (T i = 0; i < h_chain_size - 1; i++) {
        T level_span = h_chain_ptr[i + 1] - h_chain_ptr[i];
        if (level_span == 1) {
            blockPerGrid = dim3(4);
            // printf("mbsl block: %d\n", blockPerGrid.x);
            spsv_csr_n_lo_nv2011_solve_mbsl_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
                csr_row_ptr,
                csr_col_idx,
                csr_val,
                m,
                nnz,
                alpha,
                x,
                y,
                row_map,
                d_level_ptr,
                h_chain_ptr[i]
            );
        } else {
            // printf("sbml\n");
            spsv_csr_n_lo_nv2011_solve_sbml_kernel<<<1, threadPerBlock, 0, handle->stream>>>(
                csr_row_ptr,
                csr_col_idx,
                csr_val,
                m,
                nnz,
                alpha,
                x,
                y,
                row_map,
                d_level_ptr,
                h_chain_ptr[i],
                h_chain_ptr[i + 1]
            );
        }
        // cudaDeviceSynchronize();
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
