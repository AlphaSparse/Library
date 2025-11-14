#include "hip/hip_runtime.h"
#pragma once

#include "alphasparse.h"
#include "alphasparse_spmv_csr_vector.h"

#define WARPS_IN_BLOCK 4
constexpr int spmv_block_size = WARPS_IN_BLOCK * WARP_SIZE;

__device__ __forceinline__ constexpr int64_t ceildiv_d(int64_t num, int64_t den)
{
    return (num + den - 1) / den;
}

template <typename T, T warp_size>
__device__ static T lower_bound_int(const T *t, int r, int64_t target, int64_t nwarps)
{
    int l = 0;
    while (l <= r)
    {
        int m = (l + r) / 2;
        if ((int64_t)ceildivT<T>(t[m], warp_size) * nwarps < target)
        {
            l = m + 1;
        }
        else
        {
            r = m - 1;
        }
    }

    return l;
}

template <typename T, T warp_size>
__launch_bounds__(512) __global__ static void balanced_partition_row_by_nnz(const T *acc_sum_arr, T rows, T nwarps, T *partition, int64_t ave)
{
    const T gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nwarps)
        return;
    partition[gid] = lower_bound_int<T, warp_size>(acc_sum_arr, rows, (ave * gid), nwarps);
}

template <typename T, typename W, typename V, T warp_size>
__launch_bounds__(512) __global__ static void balanced_partition_row_by_nnz_and_scale_y(
    const T *acc_sum_arr,
    T rows,
    T nwarps,
    T *partition,
    int64_t ave,
    const W beta,
    V *y)
{
    const T gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= nwarps)
    {
        if (gid < nwarps + rows + 1)
        {
            y[gid - nwarps] *= beta;
        }
        return;
    }
    partition[gid] = lower_bound_int<T, warp_size>(acc_sum_arr, rows, (ave * gid), nwarps);
}

template <typename T>
__global__ static void balanced(const T *csr_row_ptr,
                                const T m,
                                T *srow,
                                T nwarps,
                                const T warp_size)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    if (nwarps > 0)
    {
        const auto num_elems = csr_row_ptr[m];
        const auto bucket_divider =
            num_elems > 0 ? ceildiv_d((int64_t)num_elems, (int64_t)warp_size) : 1;
        for (size_t i = ix; i < m; i += stride)
        {
            auto bucket =
                ceildiv_d((ceildiv_d((int64_t)csr_row_ptr[i + 1], (int64_t)warp_size) * (int64_t)nwarps),
                          bucket_divider);
            if (bucket < nwarps)
            {
                atomicAdd(&srow[bucket], 1);
            }
        }
    }
}

template <typename T>
T clac_size(const T nnz, const T warp_size, const T nwarps_)
{
    if (warp_size > 0)
    {
        int multiple = 8;
        if (nnz >= static_cast<T>(2e8))
        {
            multiple = 2048;
        }
        else if (nnz >= static_cast<T>(2e7))
        {
            multiple = 512;
        }
        else if (nnz >= static_cast<T>(2e6))
        {
            multiple = 128;
        }
        else if (nnz >= static_cast<T>(2e5))
        {
            multiple = 32;
        }
        auto nwarps = nwarps_ * multiple;
        return min(ceildivT((int64_t)nnz, (int64_t)warp_size), (int64_t)nwarps);
    }
    else
    {
        return 0;
    }
}

template <bool overflow, typename IndexType>
__device__ __forceinline__ void find_next_row(
    const IndexType num_rows,
    const IndexType nnz,
    const IndexType ind,
    IndexType &row,
    IndexType &row_end,
    const IndexType row_predict,
    const IndexType row_predict_end,
    const IndexType *__restrict__ row_ptr)
{
    if (!overflow || ind < nnz)
    {
        if (ind >= row_end)
        {
            row = row_predict;
            row_end = row_predict_end;
            if (ind < row_end)
            {
                return;
            }
            row_end = row_ptr[++row + 1];
            if (ind < row_end)
            {
                return;
            }
            row_end = row_ptr[++row + 1];
            if (ind < row_end)
            {
                return;
            }
            row_end = row_ptr[++row + 1];
            if (ind < row_end)
            {
                return;
            }

            int right = 1;
            row_end = row_ptr[row + 2];
            while (ind >= row_end)
            {
                right *= 2;
                if (row + right >= num_rows)
                {
                    right = num_rows - row - 1;
                    break;
                }
                row_end = row_ptr[row + right + 1];
            }
            if (right == 1)
            {
                ++row;
                return;
            }
            // 二分查找
            int left = right / 2;
            while (left <= right)
            {
                int mid = (right + left) / 2;
                row_end = row_ptr[row + mid + 1];
                if (ind >= row_end)
                {
                    left = mid + 1;
                }
                else
                {
                    right = mid - 1;
                }
            }
            row += left;
            row_end = row_ptr[row + 1];
            return;
            // while (ind >= row_end)
            // {
            // row_end = row_ptr[++row + 1];
            // }
        }
    }
    else
    {
        row = num_rows - 1;
        row_end = nnz;
    }
}

template <bool last, unsigned subwarp_size = 16,
          typename U, typename IndexType,
          typename V, typename Closure>
__device__ __forceinline__ void process_window(
    const cooperative_groups::thread_block_tile<subwarp_size> &group,
    const IndexType num_rows, const IndexType nnz, IndexType ind,
    IndexType &row, IndexType &row_end, IndexType &nrow, IndexType &nrow_end,
    U &temp_val, const U *val,
    const IndexType *__restrict__ col_idxs,
    const IndexType *__restrict__ csr_row_ptr, const U *b,
    V *c, Closure scale)
{
    const auto curr_row = row;
    find_next_row<last>(num_rows, nnz, ind, row, row_end, nrow, nrow_end,
                        csr_row_ptr);
    // segmented scan
    if (group.any(curr_row != row))
    {
        warp_atomic_add(group, curr_row != row, temp_val, curr_row, c, scale);
        nrow = group.shfl(row, subwarp_size - 1);
        nrow_end = group.shfl(row_end, subwarp_size - 1);
    }

    if (!last || ind < nnz)
    {
        const auto col = col_idxs[ind];
        temp_val += val[ind] * b[col];
    }
}

template <unsigned subwarp_size, typename ValueType1, typename ValueType2, typename IndexType,
          typename Closure>
__device__ __forceinline__ void warp_atomic_add(
    const cooperative_groups::thread_block_tile<subwarp_size> &group, bool force_write,
    ValueType1 &val, const IndexType row, ValueType2 *__restrict__ c,
    Closure scale)
{
    // do a local scan to avoid atomic collisions
    const bool need_write = segment_scan<subwarp_size>(
        group, row, val);
    if (need_write && force_write)
    {
        atomicAdd(&(c[row]), scale(val));
    }
    if (!need_write || force_write)
    {
        val = ValueType1{};
    }
}

template <typename IndexType>
__device__ __forceinline__ IndexType get_warp_start_idx(
    const IndexType nwarps, const IndexType nnz, const IndexType warp_idx, const IndexType warp_size)
{
    const long long cache_lines = ceildivT<IndexType>(nnz, warp_size);
    return (warp_idx * cache_lines / nwarps) * warp_size;
}

template <typename T, typename U, typename V, typename Closure>
__device__ __forceinline__ void load_balance_spmv_kernel(
    T nwarps, const T m, const T nnz,
    const U *val, const T *col_idxs,
    const T *csr_row_ptr, T *srow,
    const U *b, V *c, Closure scale,
    const T warps_in_block, const T warp_size)
{
    const T warp_idx = blockIdx.x * warps_in_block + threadIdx.y;
    if (warp_idx >= nwarps)
    {
        return;
    }
    const T start = get_warp_start_idx(nwarps, nnz, warp_idx, warp_size);
    constexpr T wsize = 32;
    const T end =
        min(get_warp_start_idx(nwarps, nnz, warp_idx + 1, warp_size),
            ceildivT<T>(nnz, wsize) * wsize);
    auto row = srow[warp_idx];
    auto row_end = csr_row_ptr[row + 1];
    auto nrow = row;
    auto nrow_end = row_end;

    U temp_val = U{};
    T ind = start + threadIdx.x;

    find_next_row<true>(m, nnz, ind, row, row_end, nrow, nrow_end,
                        csr_row_ptr);

    const T ind_end = end - wsize;
    const cooperative_groups::thread_block_tile<wsize> tile_block =
        tiled_partition<wsize>(this_thread_block());
    for (; ind < ind_end; ind += wsize)
    {
        process_window<false>(tile_block, m, nnz, ind, row,
                              row_end, nrow, nrow_end, temp_val, val, col_idxs,
                              csr_row_ptr, b, c, scale);
    }
    process_window<true>(tile_block, m, nnz, ind, row, row_end,
                         nrow, nrow_end, temp_val, val, col_idxs, csr_row_ptr, b,
                         c, scale);
    warp_atomic_add(tile_block, true, temp_val, row, c, scale);
    // V *ict_y0 = (V *)alpha_malloc(m * sizeof(V));
    // hipMemcpy(ict_y0, c, sizeof(V) * m, hipMemcpyDeviceToHost);
    // printf("\n----------------");
    // for (int i = 0; i < 20; i++)
    // {
    //     printf(",");
    // }
    // printf("--------------\n");
}

template <typename T, typename U, typename V, typename W>
__launch_bounds__(512) __global__ __launch_bounds__(spmv_block_size) void abstract_load_balance_spmv(
    T nwarps, const T m, const T nnz,
    const U *val, const T *col_idxs,
    const T *csr_row_ptr, T *srow,
    const U *b, V *c, const W alpha,
    const T warps_in_block, const T warp_size)
{
    load_balance_spmv_kernel(
        nwarps, m, nnz, val, col_idxs, csr_row_ptr, srow, b, c,
        [&alpha](const V &x)
        {
            return static_cast<V>(alpha * x);
        },
        warps_in_block, warp_size);
}

template <typename T, typename U, typename V, typename W>
static void load_balance_spmv(const T m,
                              const T n,
                              const T nnz,
                              const W alpha,
                              T *srow,
                              const T *csr_row_ptr,
                              const T *csr_col_ind,
                              const U *csr_val,
                              const U *x,
                              const W beta,
                              V *y,
                              T nwarps,
                              const T warp_size,
                              const int warps_in_block)
{
    if (nwarps > 0)
    {
        const dim3 csr_block(warp_size, warps_in_block, 1);
        const dim3 csr_grid(ceildivT((int64_t)nwarps, (int64_t)warps_in_block), 1);
        if (csr_grid.x > 0 && csr_grid.y > 0)
        {
            T *srow_device = srow;
            // hipMalloc((void **)&srow_device, nwarps * sizeof(T));
            // hipMemcpy(srow_device,
            //            srow,
            //            nwarps * sizeof(T),
            //            hipMemcpyHostToDevice);
            hipLaunchKernelGGL(abstract_load_balance_spmv, csr_grid, csr_block, 0, 0, 
                nwarps, m, nnz, csr_val,
                csr_col_ind, csr_row_ptr, srow_device, x, y,
                alpha, warps_in_block, warp_size);
            // printf("\n+++++++++++++++++++++%d,%d,%d,%d,%d,%d,%d\n", nwarps, warps_in_block, warp_size, csr_grid.x, csr_grid.y, csr_block.x, csr_block.y);
            // printf("\n+++++++++++++++++++++%d,%d,%d,%d\n", csr_grid.x, csr_grid.y, csr_block.x, csr_block.y);
        }
    }
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_load(alphasparseHandle_t handle,
                                  T m,
                                  T n,
                                  T nnz,
                                  const W alpha,
                                  const U *csr_val,
                                  const T *csr_row_ptr,
                                  const T *csr_col_ind,
                                  const U *x,
                                  const W beta,
                                  V *y,
                                  void *externalBuffer)
{
    const T SM = 80;
    const T MAX_WARP_PER_SM = 64;
    const T warp_size = WARP_SIZE;
    const T nwarps_ = SM * MAX_WARP_PER_SM / warp_size;
    const T warps_in_block = WARPS_IN_BLOCK;
    T nwarps = clac_size(nnz, warp_size, nwarps_);
    // nwarps = 204800;
    // printf("nwarps:%d\n", nwarps);
    const T BLOCK_SIZE = 512;
    // if (nwarps > m)
    // {
    //     double time1 = get_time_us();
    //     spmv_csr_vector(handle, m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind, x, beta, y);
    //     double time2 = get_time_us();
    //     printf("load balance vector spmv time: %lf\n", (time2 - time1) / (1e3));
    //     return ALPHA_SPARSE_STATUS_SUCCESS;
    // }
    // T *srow = NULL;
    // // double time1 = get_time_us();
    // hipMalloc((void **)&srow, nwarps * sizeof(T));

    // hipMemset(srow, 0, nwarps * sizeof(T));
    // auto start_time = std::chrono::high_resolution_clock::now();
    // hipLaunchKernelGGL(balanced, 64, spmv_block_size, 0, 0, csr_row_ptr, m, srow, nwarps, warp_size);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time);
    // std::cout << "balanced time: " << duration.count() << " nanoseconds" << std::endl;
    T *partition = (T *)externalBuffer;
    const int64_t ave = ceildivT<int64_t>(nnz, warp_size);
    if (beta == W{})
    {
        hipMemset(y, 0, sizeof(V) * m);
        const T GRIDSIZE = ceildivT<T>(nwarps, BLOCK_SIZE);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(balanced_partition_row_by_nnz<T, warp_size>), dim3(GRIDSIZE), dim3(BLOCK_SIZE), 0, handle->stream, 
            csr_row_ptr + 1, m - 1, nwarps, partition, ave);
    }
    else
    {
        const T GRIDSIZE = ceildivT<T>(nwarps + m, BLOCK_SIZE);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(balanced_partition_row_by_nnz_and_scale_y<T, W, V, warp_size>), dim3(GRIDSIZE), dim3(BLOCK_SIZE), 0, handle->stream, 
            csr_row_ptr + 1, m - 1, nwarps, partition, ave, beta, y);
    }
    load_balance_spmv(m, n, nnz, alpha, partition, csr_row_ptr,
                      csr_col_ind, csr_val, x, beta, y,
                      nwarps, warp_size, warps_in_block);
    hipDeviceSynchronize();

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
