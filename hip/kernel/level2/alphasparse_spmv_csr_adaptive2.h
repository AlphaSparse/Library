#include "hip/hip_runtime.h"
#include "alphasparse.h"

// CUDA kernel to find long rows
template <typename T>
__global__ void find_rows(const T *row_ptr,
                          T m,
                          T threshold1,
                          T threshold2,
                          T *nnz2,
                          T threshold3,
                          T *rows_type1,
                          T *num_rows_type1,
                          T *rows_type2,
                          T *num_rows_type2,
                          T *rows_type3,
                          T *num_rows_type3,
                          T *rows_type4,
                          T *num_rows_type4)
{
    T tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < m)
    {
        int row_length = row_ptr[tid + 1] - row_ptr[tid];
        if (row_length < threshold1)
        {
            T idx = atomicAdd(num_rows_type1, 1);
            rows_type1[idx] = tid;
        }
        else if (row_length < threshold2)
        {
            T idx = atomicAdd(num_rows_type2, 1);
            atomicAdd(nnz2, row_length);
            rows_type2[idx] = tid;
        }
        // else if (row_length < threshold3)
        // {
        //     T idx = atomicAdd(num_rows_type3, 1);
        //     rows_type3[idx] = tid;
        // }
        else
        {
            T idx = atomicAdd(num_rows_type4, 1);
            rows_type4[idx] = tid;
        }
    }
}

template <typename T, typename U, typename V, typename W>
__global__ void csrmv_kernel_block_per_row2(const T m,
                                            const W alpha,
                                            const T *csrRowPtr,
                                            const T *csrColIdx,
                                            const U *csrVal,
                                            const U *x,
                                            const W beta,
                                            V *y,
                                            const T *rows_type3,
                                            const T *num_rows_type3)
{
    int tid = threadIdx.x;
    int idx = blockIdx.x;

    __shared__ V smem[256];

    if (idx < *num_rows_type3)
    {
        T row = rows_type3[idx];
        T row_start = csrRowPtr[row];
        T row_end = csrRowPtr[row + 1];
        V sum = V{};

        for (T jj = row_start + tid; jj < row_end; jj += blockDim.x)
        {
            T col = csrColIdx[jj];
            V val = csrVal[jj];
            sum += val * x[col];
        }

        // 将每个线程的局部和存储到共享内存
        smem[tid] = sum;
        __syncthreads();

        // 使用二叉树规约方法进行求和
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (tid < stride)
            {
                smem[tid] += smem[tid + stride];
            }
            __syncthreads();
        }

        // 将最终结果写入输出向量y
        if (tid == 0)
        {
            y[row] = y[row] * beta + smem[0] * alpha;
        }
    }
}

template <int BLOCK_SIZE, int WF_SIZE, typename T, typename U, typename V, typename W>
__launch_bounds__(BLOCK_SIZE)
    __global__ static void csr_gemv_vector_memalign_ldsreduce2(const T m,
                                                               const W alpha,
                                                               const T *row_offset,
                                                               const T *csr_col_ind,
                                                               const U *csr_val,
                                                               const U *x,
                                                               const W beta,
                                                               V *y,
                                                               const int *rows_type2,
                                                               const int *num_rows_type2)
{
    const T tid = threadIdx.x;
    const T lid = threadIdx.x & (WF_SIZE - 1);        // thread index within the wavefront
    const T VECTORS_PER_BLOCK = BLOCK_SIZE / WF_SIZE; // vector num in the block
    const T vector_lane = threadIdx.x / WF_SIZE;      // vector index within the block

    T gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    T nwf = gridDim.x * BLOCK_SIZE / WF_SIZE;
    // Loop over rows
    T idx = gid / WF_SIZE;
    if (idx >= *num_rows_type2)
        return;
    int row = rows_type2[idx];
    T row_start, row_end;
    row_start = row_offset[row];
    row_end = row_offset[row + 1];
    V sum = V{};
    if (WF_SIZE == 32 && row_end - row_start > 32)
    {
        // ensure aligned memory access to csr_col_ind and csr_val
        T j = row_start - (row_start & (WF_SIZE - 1)) + lid;
        // accumulate local sums
        if (j >= row_start && j < row_end)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
        }
        // accumulate local sums
        for (j += WF_SIZE; j < row_end; j += WF_SIZE)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
        }
    }
    else
    {
        // Loop over non-zero elements
        for (T j = row_start + lid; j < row_end; j += WF_SIZE)
        {
            sum += csr_val[j] * x[csr_col_ind[j]];
        }
    }
    sum = wfreduce_sum<WF_SIZE>(sum);
    if (lid == WF_SIZE - 1)
    {
        y[row] = y[row] * beta + sum * alpha;
    }
}

#define CSRGEMV_VECTOR_ALIGN_LDSREDUCE2(WFSIZE)                    \
    {                                                              \
        dim3 csrmvn_blocks(block_num_base *WFSIZE);                \
        dim3 csrmvn_threads(BLOCK_SIZE);                           \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(csr_gemv_vector_memalign_ldsreduce2<BLOCK_SIZE, WFSIZE>), \
            csrmvn_blocks, \
            csrmvn_threads, \
            0, \
            streams[1],                                          \
            m,                                                     \
            alpha,                                                 \
            csr_row_ptr,                                           \
            csr_col_ind,                                           \
            csr_val,                                               \
            x,                                                     \
            beta,                                                  \
            y,                                                     \
            d_rows_type2,                                          \
            d_num_rows_type2);                                     \
    }

template <typename T, typename U, typename V, typename W, T UNROLL>
__global__ static void
spmv_csr_kernel2(T m,
                 T n,
                 T nnz,
                 const W alpha,
                 const U *csr_val,
                 const T *csr_row_ptr,
                 const T *csr_col_ind,
                 const U *x,
                 const W beta,
                 V *y,
                 T *rows_type1,
                 T *num_rows_type1)
{
    T idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= *num_rows_type1)
        return;
    int row = rows_type1[idx];
    T j = csr_row_ptr[row];
    y[row] = beta * y[row] + alpha * csr_val[j] * x[csr_col_ind[j]];
}

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_adaptive2(alphasparseHandle_t handle,
                                       T m,
                                       T n,
                                       T nnz,
                                       const W alpha,
                                       const U *csr_val,
                                       const T *csr_row_ptr,
                                       const T *csr_col_ind,
                                       const U *x,
                                       const W beta,
                                       V *y)
{
    printf("nnz0: %d\n", nnz);
    // Allocate memory on GPU for long_rows array and num_long_rows counter
    int *d_rows_type1;
    int *d_num_rows_type1;
    hipMalloc(&d_rows_type1, m * sizeof(int));
    hipMalloc(&d_num_rows_type1, sizeof(int));
    hipMemset(d_num_rows_type1, 0, sizeof(int));
    int *d_rows_type2;
    int *d_num_rows_type2;
    hipMalloc(&d_rows_type2, m * sizeof(int));
    hipMalloc(&d_num_rows_type2, sizeof(int));
    hipMemset(d_num_rows_type2, 0, sizeof(int));
    int *d_rows_type3;
    int *d_num_rows_type3;
    hipMalloc(&d_rows_type3, m * sizeof(int));
    hipMalloc(&d_num_rows_type3, sizeof(int));
    hipMemset(d_num_rows_type3, 0, sizeof(int));
    int *d_rows_type4;
    int *d_num_rows_type4;
    hipMalloc(&d_rows_type4, m * sizeof(int));
    hipMalloc(&d_num_rows_type4, sizeof(int));
    hipMemset(d_num_rows_type4, 0, sizeof(int));
    int *nnz2;
    hipMalloc(&nnz2, sizeof(int));
    hipMemset(nnz2, 0, sizeof(int));
    hipEvent_t event_start, event_stop;
    float elapsed_time = 0.0;

    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    // Set threshold for long rows
    const T BLOCK_SIZE = 1024;
    int threshold1 = 2;
    int threshold2 = BLOCK_SIZE;
    int threshold3 = BLOCK_SIZE;

    // Set CUDA kernel launch parameters
    int blocksPerGrid = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Call CUDA kernel to find long rows
    hipLaunchKernelGGL(find_rows, blocksPerGrid, BLOCK_SIZE, 0, 0, csr_row_ptr,
                                             m,
                                             threshold1,
                                             threshold2,
                                             nnz2,
                                             threshold3,
                                             d_rows_type1,
                                             d_num_rows_type1,
                                             d_rows_type2,
                                             d_num_rows_type2,
                                             d_rows_type3,
                                             d_num_rows_type3,
                                             d_rows_type4,
                                             d_num_rows_type4);

    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    printf("find_rows: %f\n", elapsed_time);
    int *h_num_rows_type1 = (int *)malloc(sizeof(int));
    hipMemcpy(h_num_rows_type1, d_num_rows_type1, sizeof(int), hipMemcpyDeviceToHost);
    int *h_num_rows_type2 = (int *)malloc(sizeof(int));
    hipMemcpy(h_num_rows_type2, d_num_rows_type2, sizeof(int), hipMemcpyDeviceToHost);
    int *h_num_rows_type4 = (int *)malloc(sizeof(int));
    hipMemcpy(h_num_rows_type4, d_num_rows_type4, sizeof(int), hipMemcpyDeviceToHost);
    printf("h_num_rows_type1:%d, h_num_rows_type2:%d, h_num_rows_type4:%d\n", *h_num_rows_type1, *h_num_rows_type2, *h_num_rows_type4);
    int num_streams = 3;
    hipStream_t *streams = new hipStream_t[num_streams];
    for (int i = 0; i < num_streams; ++i)
    {
        hipStreamCreate(&streams[i]);
    }
    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    int GRIDSIZE1 = (*h_num_rows_type1 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(spmv_csr_kernel2<T, U, V, W, 1>), GRIDSIZE1, BLOCK_SIZE, 0, streams[0], 
        m, n, nnz, alpha, csr_val, csr_row_ptr, csr_col_ind,
        x, beta, y, d_rows_type1, d_num_rows_type1);
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    printf("scalar: %f\n", elapsed_time);

    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    const T block_num_base = (m - 1) / BLOCK_SIZE + 1;
    const T nnz_per_row = nnz / m;
    if (nnz_per_row < 4)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE2(2);
    }
    else if (nnz_per_row < 8)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE2(4);
    }
    else if (nnz_per_row < 16)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE2(8);
    }
    else if (nnz_per_row < 32)
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE2(16);
    }
    else
    {
        CSRGEMV_VECTOR_ALIGN_LDSREDUCE2(32);
    }
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    printf("vector: %f\n", elapsed_time);

    GPU_TIMER_START(elapsed_time, event_start, event_stop);
    const T GRIDSIZE = ceildivT<T>(nnz, BLOCK_SIZE);
    hipLaunchKernelGGL(csrmv_kernel_block_per_row2, GRIDSIZE, BLOCK_SIZE, 0, streams[2], m,
                                                alpha,
                                                csr_row_ptr,
                                                csr_col_ind,
                                                csr_val,
                                                x,
                                                beta,
                                                y,
                                                d_rows_type3,
                                                d_num_rows_type3);
    // 等待所有流完成
    for (int i = 0; i < num_streams; ++i)
    {
        hipStreamSynchronize(streams[i]);
    }
    GPU_TIMER_END(elapsed_time, event_start, event_stop);
    printf("csrmv_kernel_block_per_row2: %f\n", elapsed_time);

    // 释放流资源
    for (int i = 0; i < num_streams; ++i)
    {
        hipStreamDestroy(streams[i]);
    }
    delete[] streams;
    // Free GPU memory
    hipFree(d_rows_type1);
    hipFree(d_num_rows_type1);
    hipFree(d_rows_type2);
    hipFree(d_num_rows_type2);

    return ALPHA_SPARSE_STATUS_SUCCESS;
}