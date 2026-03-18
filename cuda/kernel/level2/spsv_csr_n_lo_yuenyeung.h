#pragma once

#include "alphasparse.h"


// [row_start, row_end)
template<typename T>
static T 
get_elem_cnt_all(
    const T row_start,
    const T row_end,
    const T *row_nnz_cnt
) {
    T cnt = 0;
    for (T row = row_start; row < row_end; row++) {
        cnt += row_nnz_cnt[row];
    }
    return cnt;
}


template<typename T>
static void 
warp_divide(
    const T *row_nnz_cnt, 
    const T m, 
    const T border,
    T *len, 
    T *warp_num
) {
    const T WARP_SIZE = 32;
    warp_num[0] = 0;
    T row_end;
    T elem_cnt_all = 0;
    T k = 1;
    double elem_cnt_avg = 0;
    for (T row_start = 0; row_start < m; row_start += WARP_SIZE) {
        row_end = row_start + WARP_SIZE;          // [row_start, row_end)
        row_end = (row_end > m) ? m : row_end;
        elem_cnt_all = get_elem_cnt_all(row_start, row_end, row_nnz_cnt);
        elem_cnt_avg = (double)elem_cnt_all / (row_end - row_start);
        if (elem_cnt_avg >= border) {  // warp-level
        for (T row_cur = row_start + 1; row_cur <= row_end; row_cur++) {
            warp_num[k] = row_cur;
            k++;
        }
        } else {
            warp_num[k] = row_end;
            k++;
        }
    }
    *len = k;
    return;
}

template<typename T>
static void 
get_csr_row_nnz_cnt(
    const T *csr_row_ptr, 
    const T *csr_col_idx, 
    const T m,
    const T nnz,
    T *row_nnz_cnt
) {
    for (int row = 0; row < m; row++) {
        int cnt = 0;
        for (int ptr = csr_row_ptr[row]; ptr < csr_row_ptr[row + 1] && csr_col_idx[ptr] <= row; ptr++) {
            cnt++;
        }
        row_nnz_cnt[row] = cnt;
    }
    return;
}

template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_yuenyeung_analysis(
    alphasparseHandle_t handle,
    T m,
    T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const U* x,
    U* y,
    T** d_warp_num,
    T* warp_num_len,
    void *externalBuffer
) {
    T* h_csr_row_ptr = (T*)malloc((m + 1) * sizeof(T));
    cudaMemcpy(h_csr_row_ptr, csr_row_ptr, (m + 1) * sizeof(T), cudaMemcpyDeviceToHost);
    T* h_csr_col_idx = (T*)malloc(nnz * sizeof(T));
    cudaMemcpy(h_csr_col_idx, csr_col_idx, nnz * sizeof(T), cudaMemcpyDeviceToHost);
    T* h_row_nnz_cnt = (T*)malloc(m * sizeof(T));
    memset(h_row_nnz_cnt, 0, m * sizeof(T));
    get_csr_row_nnz_cnt(h_csr_row_ptr, h_csr_col_idx, m, nnz, h_row_nnz_cnt);
    T* h_warp_num = (T*)malloc(m * sizeof(T));
    int border = 32;
    // 计算row与warp之间的映射
    warp_divide(h_row_nnz_cnt, m, border, warp_num_len, h_warp_num);
    cudaMalloc((void**)d_warp_num, *warp_num_len * sizeof(T));
    cudaMemcpy(*d_warp_num, h_warp_num, *warp_num_len * sizeof(T), cudaMemcpyHostToDevice);
    free(h_csr_row_ptr);
    free(h_csr_col_idx);
    free(h_row_nnz_cnt);
    free(h_warp_num);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}



template<typename T, typename U>
__global__ static void
spsv_csr_n_lo_yuenyeung_kernel(
    const T* csr_row_ptr,
    const T* csr_col_idx,
    const U* csr_val,
    volatile T* get_value,
    const T m,
    const T nnz,
    const U alpha,
    const U* x,
    U* y,
    T* id_extractor,
    const T len,
    const T* d_warp_num,
    const T WARP_SIZE
) {
    const T global_id = atomicAdd(id_extractor, 1);
    const T warp_id = global_id / WARP_SIZE;
    if (warp_id >= len - 1) {
        return;
    }
    const T lane_id = (WARP_SIZE - 1) & threadIdx.x;
    T row, col;
    T ptr;
    if (d_warp_num[warp_id + 1] > d_warp_num[warp_id] + 1) {
        // thread-level
        row = d_warp_num[warp_id] + lane_id;
        if (row >= m) {
            return;
        }
        U left_sum_1 = {};
        ptr = csr_row_ptr[row];
        while (ptr < csr_row_ptr[row + 1]) {
            col = csr_col_idx[ptr];
            while (get_value[col] == 1) {
                __threadfence();//
                left_sum_1 += csr_val[ptr] * y[col];
                ptr++;
                col = csr_col_idx[ptr];
            }
            if (row == col) {
                y[row] = (alpha * x[row] - left_sum_1) / csr_val[ptr];
                __threadfence();
                get_value[row] = 1;
                return;
            }
        }
    } else {
        // warp-level
        row = d_warp_num[warp_id];
        if (row >= m) {
            return;
        }
        U left_sum_2 = {};
        for (ptr = csr_row_ptr[row] + lane_id; ptr < csr_row_ptr[row + 1]; ptr += WARP_SIZE) {
            col = csr_col_idx[ptr];
            if (col >= row) {
                break;
            }
            while (get_value[col] == 0) {
                __threadfence_block();
            }
            __threadfence();    //
            left_sum_2 += y[col] * csr_val[ptr];
        }
        for (T offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            left_sum_2 += __shfl_down_sync(0xFFFFFFFF, left_sum_2, offset, WARP_SIZE);
        }
        left_sum_2 = __shfl_sync(0xFFFFFFFF, left_sum_2, 0, WARP_SIZE);
        if (col == row && ptr < csr_row_ptr[row + 1] && col == csr_col_idx[ptr]) {
            y[row] = (alpha * x[row] - left_sum_2) / csr_val[ptr];
            __threadfence();
            get_value[row] = 1;
            return;
        }
    }
    return;
}


template<typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_yuenyeung_slove(
    alphasparseHandle_t handle,
    T m,
    T nnz,
    const U alpha,
    const U* csr_val,
    const T* csr_row_ptr,     // device variable
    const T* csr_col_idx,     // device variable
    const U* x,
    U* y,
    const T* d_warp_num,
    const T warp_num_len,
    void *externalBuffer
) { 
    const T WARP_SIZE = 32;  
  
    // for (T i = 0; i < warp_num_len; i++) {
    //   printf("warp_num[%d]: %d\n", i, h_warp_num[i]);
    // }
    
    const int threadPerBlock = 256;
    // const int threadPerBlock = 32;
    const int blockPerGrid = (warp_num_len - 2) / (threadPerBlock / WARP_SIZE) + 1;

    T *get_value = reinterpret_cast<T*>(externalBuffer);
    cudaMemset(get_value, 0, m * sizeof(T));

    T *id_extractor = reinterpret_cast<T*>(reinterpret_cast<char*>(get_value) + m * sizeof(T));
    cudaMemset(id_extractor, 0, sizeof(T));

    spsv_csr_n_lo_yuenyeung_kernel<<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
        csr_row_ptr,
        csr_col_idx,
        csr_val,
        get_value,
        m,
        nnz,
        alpha,
        x,
        y,
        id_extractor,
        warp_num_len, 
        d_warp_num,
        WARP_SIZE
    );
    return ALPHA_SPARSE_STATUS_SUCCESS;
}