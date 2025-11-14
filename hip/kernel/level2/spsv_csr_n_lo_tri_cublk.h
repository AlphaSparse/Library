#pragma once

#include "alphasparse.h"
#include "hip/hip_runtime.h"
#include <rocprim/rocprim.hpp>

#include <iomanip>

template<unsigned int WARP_SIZE, typename U>
__device__ __forceinline__ static U
warp_reduce_max(
    const U num
) {
    U tmp_num = num;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        tmp_num = max(tmp_num, __shfl_xor(tmp_num, offset));
    }
    return tmp_num;
}

template<typename T>
void
get_row_map_sorted(
    alphasparseHandle_t handle,
    const T m,
    T* done_array_i,
    T* done_array_o,
    T* row_map_i,
    T* row_map_o
) {
    unsigned int start_bit = 0;
    unsigned int end_bit = 64 - __builtin_clzll(m);
    void *temp_buffer = nullptr;
    size_t temp_size;
    rocprim::radix_sort_pairs(
        temp_buffer, 
        temp_size, 
        done_array_i, 
        done_array_o,
        row_map_i,
        row_map_o,
        m,
        start_bit,
        end_bit,
        handle->stream
    );
    hipMalloc(&temp_buffer, temp_size);
    rocprim::radix_sort_pairs(
        temp_buffer, 
        temp_size, 
        done_array_i, 
        done_array_o,
        row_map_i,
        row_map_o,
        m,
        start_bit,
        end_bit,
        handle->stream
    );
    hipFree(temp_buffer);
    return;
}

template <unsigned BLOCKSIZE, unsigned int WARP_SIZE, typename T, typename U>
__global__ static void
spsv_csr_n_lo_tri_cublk_analysis_kernel(
    const T m,
    const T *__restrict__ csr_row_ptr,
    const T *__restrict__ csr_col_idx,
	const U *__restrict__ csr_val,
    T *__restrict__ done_array,
    T *__restrict__ row_map,
	U *__restrict__ csr_rdiag)
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
		csr_rdiag[row] = U(1) / csr_val[row_end - 1];
	}
    return;
}

template <typename T>
static void __global__
get_nnz_per_row_ordered(
    const T *__restrict__ csr_row_ptr,
    const T *__restrict__ row_map,
    T m,
    T *__restrict__ nnz_per_row_ordered)
{
    T rrow = blockIdx.x * blockDim.x + threadIdx.x;
    if (rrow >= m)
    {
        return;
    }
    T row = row_map[rrow];
    nnz_per_row_ordered[rrow] = csr_row_ptr[row + 1] - csr_row_ptr[row];
}

template <typename T, typename U>
__global__ void
set_rcsr_struct(
    const T *__restrict__ csr_row_ptr,
    const T *__restrict__ csr_col_idx,
    const U *__restrict__ csr_val,
    T m,
    T nnz,
    const T *__restrict__ row_map,
    T *__restrict__ rcsr_row_ptr,
    T *__restrict__ rcsr_col_idx,
    U *__restrict__ rcsr_val,
    U *__restrict__ rcsr_rdiag)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    T rrow = tid >> 6;
    if (rrow >= m)
    {
        return;
    }
    int lane_id = tid & 63;
    T row = row_map[rrow];
    T row_begin = csr_row_ptr[row];
    T row_end = csr_row_ptr[row + 1];
    T rrow_begin = rcsr_row_ptr[rrow];
    T rrow_end = rcsr_row_ptr[rrow + 1];
    for (int i = row_begin + lane_id, j = rrow_begin + lane_id; i < row_end; i += 64, j += 64)
    {
        T col = csr_col_idx[i];
        U val = csr_val[i];
        rcsr_col_idx[j] = col;
        rcsr_val[j] = val;
        if (i == row_end - 1)
        {
            rcsr_rdiag[rrow] = U(1) / val;
        }
    }
}

template <typename T>
static void 
exclusive_scan(
    const T *d_input,
    T *d_output,
    size_t size,
    alphasparseHandle_t handle)
{
    void *d_temp_storage = nullptr;
    size_t temp_storage_size = 0;
    rocprim::exclusive_scan(
        d_temp_storage,
        temp_storage_size,
        d_input,
        d_output,
        0, // 初始值
        size,
        rocprim::plus<int>(),
        handle->stream,
        false);
    hipMalloc(&d_temp_storage, temp_storage_size);
    rocprim::exclusive_scan(
        d_temp_storage,
        temp_storage_size,
        d_input, 
        d_output,
        0, // 初始值
        size,
        rocprim::plus<int>(),
        handle->stream,
        false);
    hipFree(d_temp_storage);
}

template <typename T, typename U>
static void rearrange_csr_mat(
    const T *csr_row_ptr,
    const T *csr_col_idx,
    const U *csr_val,
    T m,
    T nnz,
    const T *row_map,
    T *rcsr_row_ptr,
    T *rcsr_col_idx,
    U *rcsr_val,
    U *rcsr_rdiag,
    alphasparseHandle_t handle)
{
    T *nnz_per_row_ordered;
    hipMalloc(&nnz_per_row_ordered, (m + 1) * sizeof(T));
    hipMemset(nnz_per_row_ordered, (T){}, (m + 1) * sizeof(T));
    dim3 threadPerBlock(256);
    dim3 blockPerGrid((m - 1) / threadPerBlock.x + 1);
    hipLaunchKernelGGL(
        get_nnz_per_row_ordered,
        blockPerGrid,
        threadPerBlock,
        0,
        handle->stream,
        csr_row_ptr,
        row_map,
        m,
        nnz_per_row_ordered);
    exclusive_scan(nnz_per_row_ordered, rcsr_row_ptr, m + 1, handle);
    blockPerGrid = dim3((m - 1) / (threadPerBlock.x / 64) + 1);
    hipLaunchKernelGGL(
        set_rcsr_struct,
        blockPerGrid,
        threadPerBlock,
        0,
        handle->stream,
        csr_row_ptr,
        csr_col_idx,
        csr_val,
        m,
        nnz,
        row_map,
        rcsr_row_ptr,
        rcsr_col_idx,
        rcsr_val,
        rcsr_rdiag);
    hipFree(nnz_per_row_ordered);
}

template <typename T>
void get_level_row_nnz_info(
	T* done_array,
	T* row_map,
	T* csr_row_ptr,
	T m,
	T nnz,
	int& level_num_out,
	double& avg_row_per_level_out,
	int& max_row_per_level_out,
	int& min_row_per_level_out,
	double& dev_row_per_level_out,
	double& avg_nnz_per_row_out,
	int& max_nnz_per_row_out,
	int& min_nnz_per_row_out,
	double& dev_nnz_per_row_out
) {
	T level_num = done_array[m - 1];
	T *row_per_level = new T[level_num];
	memset(row_per_level, 0, level_num * sizeof(T));
	for (int i = 0; i < m; i++) {
		row_per_level[done_array[i] - 1]++;
	}
	// row_per_level info
	double avg_row_per_level = 1.0 * m / level_num;
	double dev_row_per_level = 0.0;
	int max_row_per_level = row_per_level[0];
	int min_row_per_level = row_per_level[0];
	for (int i = 0; i < level_num; i++) {
		double diff = 1.0 * (row_per_level[i] - avg_row_per_level);
		dev_row_per_level += diff * diff;
		max_row_per_level = max(max_row_per_level, row_per_level[i]);
		min_row_per_level = min(min_row_per_level, row_per_level[i]);
	}
	dev_row_per_level /= level_num;
	dev_row_per_level = std::sqrt(dev_row_per_level);
	// nnz_per_row info
	double avg_nnz_per_row = 1.0 * nnz / m;
	double dev_nnz_per_row = 0.0;
	int max_nnz_per_row = csr_row_ptr[1];
	int min_nnz_per_row = csr_row_ptr[1];
	for (int i = 0; i < m; i++) {
		double cur_row_nnz = csr_row_ptr[i + 1] - csr_row_ptr[i];
		double diff = 1.0 * (cur_row_nnz - avg_nnz_per_row);
		dev_nnz_per_row += diff * diff;
		max_nnz_per_row = max(max_nnz_per_row, cur_row_nnz);
		min_nnz_per_row = min(min_nnz_per_row, cur_row_nnz);
	}
	dev_nnz_per_row /= m;
	dev_nnz_per_row = std::sqrt(dev_nnz_per_row);
	// set return
	level_num_out = level_num;
	avg_row_per_level_out = avg_row_per_level;
	max_row_per_level_out = max_row_per_level;
	min_row_per_level_out = min_row_per_level;
	dev_row_per_level_out = dev_row_per_level;
	avg_nnz_per_row_out = avg_nnz_per_row;
	max_nnz_per_row_out = max_nnz_per_row;
	min_nnz_per_row_out = min_nnz_per_row;
	dev_nnz_per_row_out = dev_nnz_per_row;
}

template <typename T>
void print_mat_row_nnz_level_info(
	T* done_array,
	T* row_map,
	T* csr_row_ptr,
	T m,
	T nnz
) {	
	T *h_done_array = new T[m];
	T *h_row_map = new T[m];
	T *h_csr_row_ptr = new T[m + 1];
	hipMemcpy(h_done_array, done_array, m * sizeof(T), hipMemcpyDeviceToHost);
	hipMemcpy(h_row_map, row_map, m * sizeof(T), hipMemcpyDeviceToHost);
	hipMemcpy(h_csr_row_ptr, csr_row_ptr, (m + 1) * sizeof(T), hipMemcpyDeviceToHost);
	int level_num;
	double avg_row_per_level;
	int max_row_per_level;
	int min_row_per_level;
	double dev_row_per_level;
	double avg_nnz_per_row;
	int max_nnz_per_row;
	int min_nnz_per_row;
	double dev_nnz_per_row;
	get_level_row_nnz_info(h_done_array, h_row_map, h_csr_row_ptr, m, nnz, 
			level_num, 
			avg_row_per_level, max_row_per_level, min_row_per_level, dev_row_per_level, 
			avg_nnz_per_row, max_nnz_per_row, min_nnz_per_row, dev_nnz_per_row);
	std::cout << std::fixed << std::setprecision(6);
	std::cout << m << "," << nnz << ",";
	std::cout << level_num << ",";
	std::cout << avg_row_per_level << "," << max_row_per_level << "," << min_row_per_level << "," << dev_row_per_level << ",";
	std::cout << avg_nnz_per_row << "," << max_nnz_per_row << "," << min_nnz_per_row << "," << dev_nnz_per_row << ",";
	exit(0);
}

template <bool REORDER, bool TOPO, typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_tri_cublk_analysis(
    alphasparseHandle_t handle,
    const T m,
    const T nnz,
    const U alpha,
    const U *csr_val,
    const T *csr_row_ptr,
    const T *csr_col_idx,
	U* csr_rdiag,
    T *row_map,
    T *rcsr_row_ptr,
    T *rcsr_col_idx,
    U *rcsr_val,
    U *rcsr_rdiag,
    const U *x,
    U *y,
    void *externalBuffer)
{
	if (!REORDER && !TOPO) {
		return ALPHA_SPARSE_STATUS_SUCCESS;
	}
    const unsigned int BLOCKSIZE = 256;
    const unsigned int WARP_SIZE = 64; // 设置为32 - 运行时死锁？
    const dim3 threadPerBlock = dim3(BLOCKSIZE);
    const dim3 blockPerGrid = dim3((m - 1) / (BLOCKSIZE / WARP_SIZE) + 1);
    T *done_array = reinterpret_cast<T *>(externalBuffer);
    hipMemset(done_array, 0, m * sizeof(T));
    hipMemset(csr_rdiag, 0, m * sizeof(T));
	hipLaunchKernelGGL(
        (spsv_csr_n_lo_tri_cublk_analysis_kernel<BLOCKSIZE, WARP_SIZE>),
        blockPerGrid,
        threadPerBlock,
        0,
        handle->stream,
        m,
        csr_row_ptr,
        csr_col_idx,
		csr_val,
        done_array,
        row_map,
		csr_rdiag);
	get_row_map_sorted(handle, m, done_array, done_array, row_map, row_map);
	if (REORDER) {
		rearrange_csr_mat(csr_row_ptr, csr_col_idx, csr_val, m, nnz, row_map, rcsr_row_ptr, rcsr_col_idx, rcsr_val, rcsr_rdiag, handle);
	}
    return ALPHA_SPARSE_STATUS_SUCCESS;
}





template <unsigned int BLOCKSIZE, unsigned int VECSIZE, bool DYNAMIC_SCHD, typename T, typename U>
__global__ static void
__launch_bounds__(BLOCKSIZE)
spsv_csr_n_lo_tri_cublk_wpr_solve_kernel_reorder(
	const T* __restrict__ rcsr_row_ptr,
	const T* __restrict__ rcsr_col_idx,
	const U* __restrict__ rcsr_val,
	const U* __restrict__ rcsr_rdiag,
	const T m,
	const T nnz,
	const U alpha,
	const U* __restrict__ x,
	volatile U* __restrict__ y,
	const T* __restrict__ row_map,
	volatile bool* __restrict__ get_value,
	T* id_extractor
) {
	T lid = threadIdx.x & 63;
	T wid = threadIdx.x >> 6;
	T stride = blockDim.x * gridDim.x / 64;
	for (T idx = blockIdx.x * (blockDim.x / 64) + wid; idx < m; idx += stride) {
		T row{}, rrow{};
		if (DYNAMIC_SCHD) {
			if (lid == 0) {
				rrow = atomicAdd(&id_extractor[0], 1);
			}
			rrow = __shfl(rrow, 0);
		} else {
			rrow = idx;
		}
		// __syncwarp();
		row = row_map[rrow];
		U rdiag = rcsr_rdiag[rrow];
		T rrow_begin = rcsr_row_ptr[rrow];
		T rrow_end = rcsr_row_ptr[rrow + 1];
		U local_sum = {};
		if (lid == 0) {
			local_sum = alpha * x[row];
		}
		T j = rrow_begin + lid;
		T local_col = m;
		U local_val = {};
		if (j < rrow_end) {
			local_col = rcsr_col_idx[j];
			local_val = rcsr_val[j];
		}
		while (j < rrow_end - 1) {
			int t = get_value[local_col];
			j += t * 64;
			if (t) {
				// local_sum -= local_val * y[local_col];
				local_sum = alphasparse_fma(-local_val, y[local_col], local_sum);
			}
			if (t && j < rrow_end - 1) {
				local_col = rcsr_col_idx[j];
				local_val = rcsr_val[j];
			}
		}
		// local_sum = vec_reduce_sum<64>(vec_id, local_sum);
		local_sum = alphasparse_wfreduce_sum<64>(local_sum);
		if (lid == 63) {
			// y[row] = local_sum / local_val;
			y[row] = local_sum * rdiag;
			__threadfence();
			get_value[row] = 1;
		}
		__syncthreads();
	}
	return;
}


template <unsigned int BLOCKSIZE, unsigned int VECSIZE, bool TOPO_SCHD, bool DYNAMIC_SCHD, typename T, typename U>
__global__ static void
__launch_bounds__(BLOCKSIZE)
spsv_csr_n_lo_tri_cublk_wpr_solve_kernel(
	const T* __restrict__ csr_row_ptr,
	const T* __restrict__ csr_col_idx,
	const U* __restrict__ csr_val,
	const U* __restrict__ csr_rdiag,
	const T m,
	const T nnz,
	const U alpha,
	const U* __restrict__ x,
	volatile U* __restrict__ y,
	const T* __restrict__ row_map,
	volatile bool* __restrict__ get_value,
	T* id_extractor
) {
	T lid = threadIdx.x & 63;
	T wid = threadIdx.x >> 6;
	T stride = blockDim.x * gridDim.x / 64;
	for (T idx = blockIdx.x * (blockDim.x / 64) + wid; idx < m; idx += stride) {
		T row{};
		if (DYNAMIC_SCHD) {
			if (lid == 0) {
				row = atomicAdd(&id_extractor[0], 1);
			}
			row = __shfl(row, 0);
		} else {
			row = idx;
		}
		if (TOPO_SCHD) {
			row = row_map[row];
			// row = alphasparse_nontemporal_load(row_map + row);
		}
		U rdiag = csr_rdiag[row];
		T row_begin = csr_row_ptr[row];
		T row_end = csr_row_ptr[row + 1];
		U local_sum = {};
		if (lid == 0) {
			local_sum = alpha * __ldg(x + row);
		}
		T j = row_begin + lid;
		T local_col = m;
		U local_val = {};
		if (j < row_end) {
			local_col = __ldg(csr_col_idx + j);
			local_val = __ldg(csr_val + j);
		}
		while (j < row_end - 1) {
			// __threadfence();
			int t = get_value[local_col];
			j += t * 64;
			if (t) {
				// __threadfence();
				local_sum = alphasparse_fma(-local_val, y[local_col], local_sum);
			}
			if (t && j < row_end - 1) {
				local_col = __ldg(csr_col_idx + j);
				local_val = __ldg(csr_val + j);
			}
		}
		local_sum = alphasparse_wfreduce_sum<64>(local_sum);
		if (lid == 63) {
			y[row] = local_sum * rdiag;
			// alphasparse_nontemporal_store(local_sum * rdiag, &y[row]);
			__threadfence();
			get_value[row] = 1;
		}
		__syncthreads();
	}
	return;
}




template <unsigned int BLOCKSIZE, unsigned int VECSIZE, typename T, typename U>
__global__ static void
__launch_bounds__(BLOCKSIZE)
spsv_csr_n_lo_tri_cublk_wpr_solve_kernel_v2(
	const T* __restrict__ rcsr_row_ptr,
	const T* __restrict__ rcsr_col_idx,
	const U* __restrict__ rcsr_val,
	const U* __restrict__ rcsr_rdiag,
	const T m,
	const T nnz,
	const U alpha,
	const U* __restrict__ x,
	volatile U* __restrict__ y,
	const T* __restrict__ row_map,
	volatile bool* __restrict__ get_value
) {
	T lid = threadIdx.x & 63;
	T wid = threadIdx.x >> 6;
	T stride = blockDim.x * gridDim.x / 64;
	for (T rrow = blockIdx.x * (blockDim.x / 64) + wid; rrow < m; rrow += stride) {
		T row = row_map[rrow];
        U rdiag = rcsr_rdiag[rrow];
		T rrow_begin = rcsr_row_ptr[rrow];
		T rrow_end = rcsr_row_ptr[rrow + 1];
		U local_sum = {};
		if (lid == 0) {
			// local_sum = alpha * x[row];
			local_sum = alpha * __ldg(x + row);
		}
		T j = rrow_begin + lid;
		T local_col = m;
		U local_val = {};
        U local_y = {};
        for (T j = rrow_begin + lid; j < rrow_end - 1; j += 64) {
            local_col = __ldg(rcsr_col_idx + j);
            local_val = __ldg(rcsr_val + j);
			int t = get_value[local_col];
            while (t == 0) {
                __threadfence();
                t = get_value[local_col];
            }
            local_y = y[local_col];
			local_sum = alphasparse_fma(-local_val, local_y, local_sum);
		}
		// local_sum = vec_reduce_sum<64>(vec_id, local_sum);
		local_sum = alphasparse_wfreduce_sum<64>(local_sum);
		if (lid == 63) {
			// y[row] = local_sum / local_val;
			y[row] = local_sum * rdiag;
			__threadfence();
			get_value[row] = 1;
		}
		__syncthreads();
	}
	return;
}

template <unsigned int BLOCKSIZE, unsigned int VECSIZE, bool DYNAMIC_SCHD, typename T, typename U>
__global__ static void
__launch_bounds__(BLOCKSIZE)
spsv_csr_n_lo_tri_cublk_tpr_solve_kernel_reorder(
	const T* __restrict__ rcsr_row_ptr,
	const T* __restrict__ rcsr_col_idx,
	const U* __restrict__ rcsr_val,
	const U* __restrict__ rcsr_rdiag,
	const T m,
	const T nnz,
	const U alpha,
	const U* __restrict__ x,
	volatile U* __restrict__ y,
	const T* __restrict__ row_map,
	volatile bool* __restrict__ get_value,
	T* id_extractor
) {
	T stride = blockDim.x * gridDim.x;
	for (T idx = blockIdx.x * blockDim.x + threadIdx.x; idx < m; idx += stride) {
		T row{}, rrow{};
		if (DYNAMIC_SCHD) {
			rrow = atomicAdd(&id_extractor[0], 1);
		} else {
			rrow = idx;
		}
		row = row_map[rrow];
		U rdiag = rcsr_rdiag[rrow];
		T rptr = rcsr_row_ptr[rrow];
		T rrow_end = rcsr_row_ptr[rrow + 1];
		U tmp_sum = alpha * x[row];
		T col_id = rcsr_col_idx[rptr];
		bool flag = true;
		while (flag && rptr < rrow_end) {
			while (get_value[col_id] == 1) {
				// __threadfence();
				tmp_sum = alphasparse_fma(-y[col_id], rcsr_val[rptr], tmp_sum);
				rptr++;
				col_id = rcsr_col_idx[rptr];
			}
			if (col_id == row) {
				y[row] = tmp_sum * rdiag;
				__threadfence();
				get_value[row] = 1;
				flag = false;
			}
		}
		__syncthreads();
	}
	return;
}


template <unsigned int BLOCKSIZE, unsigned int VECSIZE, bool TOPO_SCHD, bool DYNAMIC_SCHD, typename T, typename U>
__global__ static void
__launch_bounds__(BLOCKSIZE)
spsv_csr_n_lo_tri_cublk_tpr_solve_kernel(
	const T* __restrict__ csr_row_ptr,
	const T* __restrict__ csr_col_idx,
	const U* __restrict__ csr_val,
	const U* __restrict__ csr_rdiag,
	const T m,
	const T nnz,
	const U alpha,
	const U* __restrict__ x,
	volatile U* __restrict__ y,
	const T* __restrict__ row_map,
	volatile bool* __restrict__ get_value,
	T* id_extractor
) {
	T stride = blockDim.x * gridDim.x;
	for (T idx = blockIdx.x * blockDim.x + threadIdx.x; idx < m; idx += stride) {
		T row_id{};
		if (DYNAMIC_SCHD) {
			row_id = atomicAdd(&id_extractor[0], 1);
		} else {
			row_id = idx;
		}
		if (TOPO_SCHD) {
			row_id = row_map[row_id];
		}
		U rdiag = csr_rdiag[row_id];
		T ptr = csr_row_ptr[row_id];
		T row_end = csr_row_ptr[row_id + 1];
		U tmp_sum = alpha * x[row_id];
		T col_id = csr_col_idx[ptr];
		bool flag = true;
		while (flag && ptr < row_end) {
			while (get_value[col_id] == 1) {
				// __threadfence();
				tmp_sum = alphasparse_fma(-y[col_id], csr_val[ptr], tmp_sum);
				ptr++;
				col_id = csr_col_idx[ptr];
			}
			if (col_id == row_id) {
				y[row_id] = tmp_sum * rdiag;
				__threadfence();
				get_value[row_id] = 1;
				flag = false;
			}
		}
		__syncthreads();
	}
	return;
}



template <bool REORDER, bool TOPO_SCHD, bool DYNAMIC_SCHD, bool WPR, typename T, typename U>
alphasparseStatus_t
spsv_csr_n_lo_tri_cublk_solve(
	alphasparseHandle_t handle,
	const T m,
	const T nnz,
	const U alpha,
	const U* csr_val,
	const T* csr_row_ptr,
	const T* csr_col_idx,
	const U* csr_rdiag,
	const U* rcsr_val,
	const T* rcsr_row_ptr,
	const T* rcsr_col_idx,
    const U* rcsr_rdiag,
	T* row_map,
	const U* x,
	U* y,
	void *externalBuffer
) {
	/*
	T* h_row_map = new T[m];
	hipMemcpy(h_row_map, row_map, m * sizeof(T), hipMemcpyDeviceToHost);
	for (int i = 0; i < m; i++) {
		std::cout << i << ": " << h_row_map[i] << std::endl;
	}
	exit(0);
	*/
	// printf("spsv tri cublk\n");
	const unsigned int VECSIZE = 64;	// 设置为32时卡死
	// constexpr unsigned int cu_num = 64 * 8;
	// constexpr unsigned int cu_num = 64;
	// constexpr unsigned int cu_num = 128;
	// constexpr unsigned int cu_num = 256;
	constexpr unsigned int cu_num = 256;
	const unsigned int BLOCKSIZE = 256;
	dim3 threadPerBlock(BLOCKSIZE);
	dim3 blockPerGrid(max(cu_num, (m - 1) / (BLOCKSIZE / VECSIZE) + 1));
	// dim3 blockPerGrid(cu_num);
	bool* get_value = reinterpret_cast<bool*>(externalBuffer);
	hipMemset(get_value, 0, m * sizeof(bool));
	//KERNEL_DISPATCH(BLOCKSIZE, 16);
	// printf("blockPerGrid %d\n", blockPerGrid);
	T* id_extractor = reinterpret_cast<T*>(reinterpret_cast<char*>(get_value) + sizeof(T) * m);
	hipMemset(id_extractor, 0, sizeof(T));
	/*
	WPR ? printf("WPR\n") : printf("TPR\n");
	REORDER ? printf("REORDER\n") : printf("NO-REORDER\n");
	TOPO_SCHD ? printf("TOPO_SCHD\n") : printf("NO-TOPO_SCHD\n");
	DYNAMIC_SCHD ? printf("DYNAMIC_SCHD\n") : printf("STATIC_SCHD\n");
	*/
	if (WPR) {
		if (REORDER) {
			// printf("spsv_csr_n_lo_tri_cublk_wpr_solve_kernel_reorder\n");
		hipLaunchKernelGGL(
			(spsv_csr_n_lo_tri_cublk_wpr_solve_kernel_reorder<BLOCKSIZE, VECSIZE, DYNAMIC_SCHD>),
				blockPerGrid, 
				threadPerBlock, 
				0, 
				handle->stream,
		        rcsr_row_ptr,
				rcsr_col_idx,
			    rcsr_val,
		        rcsr_rdiag,
		        m, 
				nnz,
			    alpha,
				x,
		        y,
			    row_map,
		        get_value,
				id_extractor
			);
		} else {
			// printf("spsv_csr_n_lo_tri_cublk_wpr_solve_kernel\n");
		hipLaunchKernelGGL(
			(spsv_csr_n_lo_tri_cublk_wpr_solve_kernel<BLOCKSIZE, VECSIZE, TOPO_SCHD, DYNAMIC_SCHD>),
				blockPerGrid,
				threadPerBlock,
				0,
				handle->stream,
				csr_row_ptr,
				csr_col_idx,
				csr_val,
				csr_rdiag,
				m,
				nnz,
				alpha,
				x,
				y,
				row_map,
				get_value,
				id_extractor
			);
		}
	} else {
		blockPerGrid = dim3(min(cu_num, (m - 1) / BLOCKSIZE + 1));
		if (REORDER) {
			// printf("spsv_csr_n_lo_tri_cublk_tpr_solve_kernel_reorder\n");
		hipLaunchKernelGGL(
			(spsv_csr_n_lo_tri_cublk_tpr_solve_kernel_reorder<BLOCKSIZE, VECSIZE, DYNAMIC_SCHD>),
				blockPerGrid, 
				threadPerBlock, 
				0, 
				handle->stream,
		        rcsr_row_ptr,
				rcsr_col_idx,
			    rcsr_val,
		        rcsr_rdiag,
		        m, 
				nnz,
			    alpha,
				x,
		        y,
			    row_map,
		        get_value,
				id_extractor
			);
		
		} else {
			// printf("spsv_csr_n_lo_tri_cublk_tpr_solve_kernel\n");
		hipLaunchKernelGGL(
			(spsv_csr_n_lo_tri_cublk_tpr_solve_kernel<BLOCKSIZE, VECSIZE, TOPO_SCHD, DYNAMIC_SCHD>),
				blockPerGrid,
				threadPerBlock,
				0,
				handle->stream,
				csr_row_ptr,
				csr_col_idx,
				csr_val,
				csr_rdiag,
				m,
				nnz,
				alpha,
				x,
				y,
				row_map,
				get_value,
				id_extractor
			);
		}

	}

	return ALPHA_SPARSE_STATUS_SUCCESS;
}
