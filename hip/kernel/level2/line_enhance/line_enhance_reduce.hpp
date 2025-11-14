#ifndef SPMV_ACC_LINE_ENHANCE_REDUCE_HPP
#define SPMV_ACC_LINE_ENHANCE_REDUCE_HPP

constexpr int LE_REDUCE_OPTION_DIRECT = 0;
constexpr int LE_REDUCE_OPTION_VEC = 1;
constexpr int LE_REDUCE_OPTION_VEC_MEM_COALESCING = 2;

constexpr int DEFAULT_LE_REDUCE_OPTION = LE_REDUCE_OPTION_DIRECT;

/**
 * direct reduction method: each thread perform reduction for one row in matrix.
 * @tparam I type of index
 * @tparam T type of float number data
 * @param reduce_row_id the row id for reduction for current thread.
 * @param block_row_end the end row id (not include) of this HIP block in calculation
 * @param reduce_row_idx_begin row_offset[\param reduce_row_id]
 * @param reduce_row_idx_end row_offset[\param reduce_row_id + 1]
 * @param block_round_inx_start start index of matrix values of current round in current HIP block.
 * @param block_round_inx_end end index of matrix values of current round in current HIP block.
 * @param shared_val LDS address
 * @param sum partial sum
 * @return
 */
template <typename I, typename T>
__device__ __forceinline__ void line_enhance_direct_reduce(const I reduce_row_id, const I block_row_end,
                                                           const I reduce_row_idx_begin, const I reduce_row_idx_end,
                                                           const I block_round_inx_start, const I block_round_inx_end,
                                                           const T *shared_val, T &sum)
{
  if (reduce_row_id < block_row_end)
  {
    if (reduce_row_idx_begin < block_round_inx_end && reduce_row_idx_end > block_round_inx_start)
    {
      const I reduce_start = max(reduce_row_idx_begin, block_round_inx_start);
      const I reduce_end = min(reduce_row_idx_end, block_round_inx_end);
      for (I j = reduce_start; j < reduce_end; j++)
      {
        sum += shared_val[j - block_round_inx_start];
      }
    }
  }
}

/**
 * direct and vector reduction method: each vector perform reduction for one row in matrix.
 * If the vector size is set to 1, it means one thread in block reduce one row.
 * @tparam I type of index
 * @tparam T type of float number data
 * @tparam VECTOR_SIZE threads number in a vector for reduction.
 * @oaram: tid_in_vec thread id in vector.
 * @note: other parameter keep the same as device function line_enhance_direct_reduce.
 */
template <typename I, typename T, int VECTOR_SIZE>
__device__ __forceinline__ T line_enhance_vec_reduce(const I reduce_row_id, const I block_row_end,
                                                     const I reduce_row_idx_begin, const I reduce_row_idx_end,
                                                     const I block_round_inx_start, const I block_round_inx_end,
                                                     const T *shared_val, const int tid_in_vec)
{
  T local_sum = T{};
  if (reduce_row_id < block_row_end)
  {
    if (reduce_row_idx_begin < block_round_inx_end && reduce_row_idx_end > block_round_inx_start)
    {
      // (label-1)
      // reduce from LDS
      const I reduce_start = max(reduce_row_idx_begin, block_round_inx_start);
      const I reduce_end = min(reduce_row_idx_end, block_round_inx_end);
      for (I j = reduce_start + tid_in_vec; j < reduce_end; j += VECTOR_SIZE)
      {
        local_sum += shared_val[j - block_round_inx_start];
      }
    }
  }
  return local_sum;
}

// local shift can reduce data inside a vector to the first thread in the vector.
template <typename I, typename T, int VECTOR_SIZE>
__device__ __forceinline__ void line_enhance_vec_local_shift(T &data)
{
  cooperative_groups::thread_block g = cooperative_groups::this_thread_block();
  const cooperative_groups::thread_block_tile<VECTOR_SIZE> tile_block =
      cooperative_groups::tiled_partition<VECTOR_SIZE>(g);

  // reduce from lanes in a vector
  if (VECTOR_SIZE > 1)
  { // in fact, this branch is unnecessary.
#pragma unroll
    for (int i = VECTOR_SIZE >> 1; i > 0; i >>= 1)
    {
      data += tile_block.shfl_down(data, i);
      // or use: __shfl_down(i, VECTOR_SIZE);
    }
  }
}

// global shift moves data from the first thread inside vectors to the front threads inside a block.
// It can make y storing memory-coalescing in vector-based reduction step.
// note: please make sure: 1. the shared_val array is large enough (large then vector number in block).
// 2. vector number must less or equal than the rows processed by a block.
template <typename I, typename T, int VECTORS_NUM>
__device__ __forceinline__ T line_enhance_vec_global_shift(const I tid_in_block, const I vec_id_in_block,
                                                           const I tid_in_vec, T *shared_val, const T data)
{
  __syncthreads(); // sync previous LDS reading in vector reduction step.
  if (tid_in_vec == 0)
  {
    shared_val[vec_id_in_block] = data;
  }
  __syncthreads();
  T shift_data = T{};
  if (tid_in_block < VECTORS_NUM)
  {
    shift_data = shared_val[tid_in_block];
  }
  return shift_data;
}

#endif // SPMV_ACC_LINE_ENHANCE_REDUCE_HPP
