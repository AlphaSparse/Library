#pragma once

#include "alphasparse.h"

// alg_num: 16

/*
    把获取csr_row_idx的步骤从solve阶段移到analysis里
*/

// Preprocessing phase.
// Calculate the dependence of each row for a lower left matrix in CSR format,
// and store every NNZ's row index in csr_row_idx.
// One thread processes one row.
template<typename T>
__global__ static void
spsv_csr_n_lo_nnz_balance_preprocess(
    const T* csr_row_ptr,   /* <I> */
    const T* csr_col_idx,   /* <I> */
    const T m,              /* <I> */
    T* in_degree,           /* <O> */
    T* csr_row_idx,         /* <O> Similar to coo_row_idx.
                                Length: nnz. */
    const T WARP_SIZE
) {
    T tid = blockIdx.x * blockDim.x + threadIdx.x;
    T row_id = tid / WARP_SIZE;
    if (row_id >= m) {
        return;
    }
    T lane_id = tid % WARP_SIZE;
    T cnt = 0;
    for (T ptr = csr_row_ptr[row_id] + lane_id; ptr < csr_row_ptr[row_id + 1] && row_id >= csr_col_idx[ptr]; ptr += WARP_SIZE) {
        cnt++;
        csr_row_idx[ptr] = row_id;
    }
    atomicAdd(&in_degree[row_id], cnt);
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_nnz_balance_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    T* csr_row_idx,
    T* in_degree,
    const U* x,
    U* y,
    void* externalBuffer,
    bool is_conj = false
) {
    int threadPerBlock = 256;
    int blockPerGrid;
    const int WARP_SIZE = 32;

    blockPerGrid = (m * WARP_SIZE - 1) / threadPerBlock + 1;

    cudaMemset(csr_row_idx, 0, nnz * sizeof(T));
    cudaMemset(in_degree, 0, m * sizeof(T));
    // printf("nnz: %d\n", nnz);
    spsv_csr_n_lo_nnz_balance_preprocess<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        csr_row_ptr,
        csr_col_idx,
        m,
        in_degree,
        csr_row_idx,
        WARP_SIZE
    );

    return ALPHA_SPARSE_STATUS_SUCCESS;
}


template<typename T, typename U>
__global__ static void
spsv_csr_n_lo_nnz_balance_solve_kernel(
    const T* csr_row_idx,
    const T* csr_col_idx,
    const U* csr_val,
    const T nnz,
    const U alpha,
    const U* x,
    U* y,
    volatile T* get_value,
    U* tmp_sum,
    T* in_degree
) {
    T val_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (val_id >= nnz) {
        return;
    }
    T row_id = csr_row_idx[val_id];
    T col_id = csr_col_idx[val_id];
    if (row_id < col_id) {
        return;
    }
    while(true) {
        if (row_id != col_id) {
            if (get_value[col_id] == 1) {
                atomicAdd(&tmp_sum[row_id], y[col_id] * csr_val[val_id]);
                __threadfence();
                atomicSub(&in_degree[row_id], 1);
                return;
            }
        } else {
            __threadfence();
            if (in_degree[row_id] == 1) {
                y[col_id] = (alpha * x[col_id] - tmp_sum[row_id]) / csr_val[val_id];
                __threadfence();
                get_value[col_id] = 1;
                return;
            }
        }
    }
    // printf("%d, %d, %d\n", val_id, csr_row_idx[val_id], in_degree[csr_row_idx[val_id]]);
    return;
}


template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_nnz_balance_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_idx,
    const T* csr_col_idx,
    const T* in_degree,
    const U* x,
    U* y,
    void* externalBuffer
) {
    int threadPerBlock = 256;
    int blockPerGrid = (nnz - 1) / threadPerBlock + 1;

    U *tmp_sum = reinterpret_cast<U*>(externalBuffer);
    cudaMemset(tmp_sum, 0, m * sizeof(U));

    T *get_value = reinterpret_cast<T*>(reinterpret_cast<char*>(tmp_sum) + m * sizeof(U));
    cudaMemset(get_value, 0, m * sizeof(T));

    T *tmp_in_degree = reinterpret_cast<T*>(reinterpret_cast<char*>(get_value) + m * sizeof(T));
    cudaMemcpy(tmp_in_degree, in_degree, m * sizeof(T), cudaMemcpyDeviceToDevice);

    // U* h_tmp_sum = (U*)malloc(m * sizeof(U));
    // cudaMemcpy(h_tmp_sum, tmp_sum, m * sizeof(U), cudaMemcpyDeviceToHost);
    // T* h_get_value = (T*)malloc(m * sizeof(T));
    // cudaMemcpy(h_get_value, get_value, m * sizeof(T), cudaMemcpyDeviceToHost);
    // for (int i = 0; i < m; i++) {
    //     printf("%.4lf, %d\n", h_tmp_sum[i], h_get_value[i]);
    // }

    spsv_csr_n_lo_nnz_balance_solve_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        csr_row_idx,
        csr_col_idx,
        csr_val,
        nnz,
        alpha,
        x,
        y,
        get_value,
        tmp_sum,
        tmp_in_degree
    );
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}