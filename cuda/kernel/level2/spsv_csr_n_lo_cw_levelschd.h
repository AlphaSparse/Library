#pragma once

#include "alphasparse.h"
#include "alphasparse/types.h" 

template<typename T, typename U>
__global__ static void
spsv_csr_n_lo_cw_levelschd_kernel_volatile(
    const T* __restrict__ csr_row_ptr,
    const T* __restrict__ csr_col_idx,
    const U* __restrict__ csr_val,
    const T* __restrict__ row_map,
    const T m,
    const U alpha,
    const U* __restrict__ x,
    volatile U* __restrict__ y,
    volatile T* __restrict__ get_value,
    T* id_extractor
) {
    T row_id = atomicAdd(id_extractor, 1);
    if (row_id >= m) {
        return;
    }
    row_id = row_map[row_id];
    T ptr = csr_row_ptr[row_id];
    U tmp_sum = {};
    T col_id = csr_col_idx[ptr];
    while (ptr < csr_row_ptr[row_id + 1]) {
        while (get_value[col_id] == 1) {
            tmp_sum = tmp_sum + y[col_id] * csr_val[ptr];
            ptr++;
            col_id = csr_col_idx[ptr];
        }
        if (col_id == row_id) {
            y[row_id] = (alpha * x[row_id] - tmp_sum) / csr_val[ptr];
            __threadfence();    
            get_value[row_id] = 1;
            return;
        }
    }
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_cw_levelschd_kernel_volatile(
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const cuFloatComplex* csr_val,
    const T* row_map,
    const T m,
    const cuFloatComplex alpha,
    const cuFloatComplex* x,
    volatile cuFloatComplex* y,
    volatile T* get_value,
    T* id_extractor
) {
    return;
}

template<typename T>
__global__ static void
spsv_csr_n_lo_cw_levelschd_kernel_volatile(
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const cuDoubleComplex* csr_val,
    const T* row_map,
    const T m,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* x,
    volatile cuDoubleComplex* y,
    volatile T* get_value,
    T* id_extractor
) {
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_cw_levelschd_solve(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const T* row_map,
    const U* x,
    U* y,
    void *externalBuffer
) {
    const int threadPerBlock = 256;
    const int blockPerGrid = (m - 1) / threadPerBlock + 1;

    // get_value mem: sizeof(T) * m
    T *get_value = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(get_value, 0, m * sizeof(T));
    // id_extractor mem: sizeof(T) * 1
    T *id_extractor = reinterpret_cast<T*>(reinterpret_cast<char*>(get_value) + sizeof(T) * m);
    cudaMemset(id_extractor, 0, sizeof(T));

    spsv_csr_n_lo_cw_levelschd_kernel_volatile<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        csr_row_ptr,
        csr_col_idx,
        csr_val,
        row_map,
        m,
        alpha,
        x,
        y,
        get_value,
        id_extractor
    );
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
