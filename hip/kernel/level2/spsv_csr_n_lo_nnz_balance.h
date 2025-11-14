#pragma once

#include "alphasparse.h"

template <unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T>
__global__ static void
spsv_csr_n_lo_nnz_balance_analysis_kernel_topo(
    const T m,
    const T *__restrict__ csr_row_ptr,
    const T *__restrict__ csr_col_idx,
    T *__restrict__ done_array,
    T *__restrict__ row_map)
{
    T lid = threadIdx.x & (WARP_SIZE - 1);
    T wid = threadIdx.x / WARP_SIZE;
    T first_row = blockIdx.x * (blockDim.x / WARP_SIZE);
    T row = first_row + wid;
    __shared__ T local_done_array[BLOCKSIZE / WARP_SIZE];
    if (lid == 0)
    {
        local_done_array[wid] = 0;
    }
    __syncthreads();
    if (row >= m)
    {
        return;
    }
    if (lid == 0)
    {
        row_map[row] = row;
    }
    T local_max = 0;
    T row_begin = csr_row_ptr[row];
    T row_end = csr_row_ptr[row + 1];
    T j = row_begin + lid;
    T local_col = csr_col_idx[j];
    while (j < row_end && local_col < first_row)
    {
        __threadfence();
        T local_done = done_array[local_col];
        local_max = max(local_done, local_max);
        j += (local_done != 0) * WARP_SIZE;
        if (local_done != 0 && j < row_end)
        {
            local_col = csr_col_idx[j];
        }
    }
    while (j < row_end && local_col < row)
    {
        T local_idx = local_col - first_row;
        __threadfence_block();
        T local_done = local_done_array[local_idx];
        j += (local_done != 0) * WARP_SIZE;
        local_max = max(local_done, local_max);
    }
    local_max = warp_reduce_max<WARP_SIZE>(local_max);
    if (lid == WARP_SIZE - 1)
    {
        __hip_atomic_store(&local_done_array[wid], local_max + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
        done_array[row] = local_max + 1;
        __threadfence();
    }
    return;
}

template <bool REORDER, typename T>
__global__ static void
spsv_csr_n_lo_nnz_balance_analysis_kernel(
	const T *csr_row_ptr,
	const T *csr_col_idx,
	const T* row_map,
	const T m,
	T *in_degree,
	T *csr_row_idx)
{
	T tid = threadIdx.x + blockIdx.x * blockDim.x;
	T row_id = tid >> 6;
	T origin_row_id = row_id;
	if (row_id >= m)
	{
		return;
	}
	if (REORDER) {
		origin_row_id = row_map[row_id];
	}
	T lane_id = tid & 63;
	T cnt = 0;
	for (T ptr = csr_row_ptr[row_id] + lane_id; ptr < csr_row_ptr[row_id + 1] && origin_row_id >= csr_col_idx[ptr]; ptr += 64)
	{
		cnt++;
		csr_row_idx[ptr] = origin_row_id;
	}
	alphasparse_wfreduce_sum<64>(&cnt);
	if (lane_id == 63)
	{
		in_degree[origin_row_id] = cnt;
	}
}

template <bool REORDER, typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_nnz_balance_analysis(
	alphasparseHandle_t handle,
	const T m,
	const T nnz,
	const U alpha,
	const T *csr_row_ptr,
	const T *csr_col_idx,
	const U *csr_val,
	T *csr_row_idx,
	T *in_degree,
	T* row_map,
	T* rcsr_row_ptr,
	T* rcsr_col_idx,
	U* rcsr_val,
	U* rcsr_rdiag,
	const U *x,
	U *y,
	void *externalBuffer)
{
	dim3 threadPerBlock;
	dim3 blockPerGrid;

	if (REORDER) {
		const unsigned int BLOCKSIZE = 256;
    	const unsigned int WARP_SIZE = 64;
    	threadPerBlock = dim3(BLOCKSIZE);
    	blockPerGrid = dim3((m - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
    	T *done_array = reinterpret_cast<T *>(externalBuffer);
		hipMemset(done_array, 0, m * sizeof(T));
		hipLaunchKernelGGL(
			(spsv_csr_n_lo_nnz_balance_analysis_kernel_topo<BLOCKSIZE, WARP_SIZE>),
			blockPerGrid, 
			threadPerBlock, 
			0, 
			handle->stream,
			m,
			csr_row_ptr,
			csr_col_idx,
			done_array,
			row_map);
		get_row_map_sorted(handle, m, done_array, done_array, row_map, row_map);
    
		rearrange_csr_mat(
    		csr_row_ptr,
    		csr_col_idx,
    		csr_val,
    		m,
    		nnz,
    		row_map,
    		rcsr_row_ptr,
    		rcsr_col_idx,
    		rcsr_val,
    		rcsr_rdiag,
    		handle);

		threadPerBlock = dim3(256);
		blockPerGrid = dim3((m - 1) / (threadPerBlock.x / 64) + 1);
		hipLaunchKernelGGL(
			spsv_csr_n_lo_nnz_balance_analysis_kernel<REORDER>,
			blockPerGrid,
			threadPerBlock,
			0,
			handle->stream,
			rcsr_row_ptr,
			rcsr_col_idx,
			row_map,
			m,
			in_degree,
			csr_row_idx);
	} else {
		threadPerBlock = dim3(256);
		blockPerGrid = dim3((m - 1) / (threadPerBlock.x / 64) + 1);

		hipLaunchKernelGGL(
			spsv_csr_n_lo_nnz_balance_analysis_kernel<REORDER>,
			blockPerGrid, 
			threadPerBlock,
			0,
			handle->stream,
			csr_row_ptr,
			csr_col_idx,
			row_map,
			m,
			in_degree,
			csr_row_idx);
	}
	return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <bool REORDER, unsigned int BLOCKSIZE, typename T, typename U>
__global__ static void
__launch_bounds__(BLOCKSIZE)
	spsv_csr_n_lo_nnz_balance_solve_kernel(
		const T *__restrict__ csr_row_idx,
		const T *__restrict__ csr_col_idx,
		const U *__restrict__ csr_val,
		const T* __restrict__ row_map,
		const T nnz,
		const U alpha,
		const U *__restrict__ x,
		U *y,
		volatile T *__restrict__ get_value,
		U *__restrict__ tmp_sum,
		T *__restrict__ in_degree)
{
	T val_id = threadIdx.x + blockIdx.x * blockDim.x;
	if (val_id >= nnz)
	{
		return;
	}
	T row_id = csr_row_idx[val_id];
	T col_id = csr_col_idx[val_id];
	if (row_id < col_id)
	{
		return;
	}
	int done = 0;
	U val = csr_val[val_id];
	U xx = x[col_id];
	U diag = (U)1 / val;
	while (done == 0)
	{
		if (row_id != col_id)
		{
			if (get_value[col_id] == 1)
			{
				__threadfence();
				atomicAdd(&tmp_sum[row_id], y[col_id] * val);
				__threadfence();
				atomicSub(&in_degree[row_id], 1);
				done = 1;
			}
		}
		if (row_id == col_id)
		{
			__threadfence();
			if (in_degree[row_id] == 1)
			{
				__threadfence();
				y[col_id] = (alpha * xx - tmp_sum[row_id]) * diag;
				__threadfence();
				get_value[col_id] = 1;
				done = 1;
			}
		}
	}
	return;
}

template <bool REORDER, typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_nnz_balance_solve(
	alphasparseHandle_t handle,
	const T m,
	const T nnz,
	const U alpha,
	const U *csr_val,
	const T *csr_row_idx,
	const T *csr_col_idx,
	const T *row_map,
	const T *in_degree,
	const U *x,
	U *y,
	void *externalBuffer)
{
	const unsigned int BLOCKSIZE = 256;
	dim3 threadPerBlock(BLOCKSIZE);
	dim3 blockPerGrid((nnz - 1) / BLOCKSIZE + 1);

	U *tmp_sum = reinterpret_cast<U *>(externalBuffer);
	hipMemset(tmp_sum, 0, m * sizeof(U));

	T *get_value = reinterpret_cast<T *>(reinterpret_cast<char *>(tmp_sum) + m * sizeof(U));
	hipMemset(get_value, 0, m * sizeof(T));

	T *tmp_in_degree = reinterpret_cast<T *>(reinterpret_cast<char *>(get_value) + m * sizeof(T));
	hipMemcpy(tmp_in_degree, in_degree, m * sizeof(T), hipMemcpyDeviceToDevice);
	spsv_csr_n_lo_nnz_balance_solve_kernel<REORDER, BLOCKSIZE><<<blockPerGrid, threadPerBlock, 0, handle->stream>>>(
		csr_row_idx,
		csr_col_idx,
		csr_val,
		row_map,
		nnz,
		alpha,
		x,
		y,
		get_value,
		tmp_sum,
		tmp_in_degree);
	return ALPHA_SPARSE_STATUS_SUCCESS;
}
