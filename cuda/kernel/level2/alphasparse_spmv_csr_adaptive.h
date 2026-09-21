#pragma once

#include "alphasparse.h"
#include "alphasparse/types.h" 
#include "alphasparse/common.h"

#define BLOCK_SIZE 1024
#define BLOCK_MULTIPLIER 3
#define ROWS_FOR_VECTOR 1
#define WG_BITS 24
#define ROW_BITS 32
#define WG_SIZE 256

/**
 * csr-adaptive
 *
 * Greathouse J L, Daga M. Efficient sparse matrix-vector multiplication on GPUs using the CSR storage format[C]//SC'14
 * Daga M, Greathouse J L. Structural agnostic SpMV: Adapting CSR-adaptive for irregular matrices[C]//HiPC'15
 *
 */

template <typename T, typename U, typename V, typename W>
__launch_bounds__(WG_SIZE)
    __global__ void csrmvn_adaptive_device(unsigned long long *row_blocks,
                                           W alpha,
                                           T nnz,
                                           const T *csr_row_ptr,
                                           const T *csr_col_ind,
                                           const U *csr_val,
                                           const U *x,
                                           W beta,
                                           V *y)
{
  __shared__ V partialSums[BLOCK_SIZE];

  T lid = threadIdx.x;
  T gid = blockIdx.x;

  // The row blocks buffer holds a packed set of information used to inform each
  // workgroup about how to do its work:
  //
  // |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
  // |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
  // |------------Row Information------------|--------^|---WG ID within a long row---|
  // |                                       |    flag/|or # reduce threads for short|
  //
  // The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
  // row it will be working on. When one workgroup calculates multiple rows, this
  // rowBlock entry and the next one tell it the range of rows to work on.
  // The lower 24 bits are used whenever multiple workgroups calculate a single long
  // row. This tells each workgroup its ID within that row, so it knows which
  // part of the row to operate on.
  // Alternately, on short row blocks, the lower bits are used to communicate
  // the number of threads that should be used for the reduction. Pre-calculating
  // this on the CPU-side results in a noticable performance uplift on many matrices.
  // Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
  // know when the first workgroup for that row has finished initializing the output
  // value. While this bit is the same as the first workgroup's flag bit, this
  // workgroup will spin-loop.
  T row = ((row_blocks[gid] >> (64 - ROW_BITS)) & ((1ULL << ROW_BITS) - 1ULL));
  T stop_row = ((row_blocks[gid + 1] >> (64 - ROW_BITS)) & ((1ULL << ROW_BITS) - 1ULL));
  T num_rows = stop_row - row;

  // Get the workgroup within this long row ID out of the bottom bits of the row block.
  T wg = row_blocks[gid] & ((1 << WG_BITS) - 1);

  const T row_offset = csr_row_ptr[row];

  V temp_sum = {};

  // If the next row block starts more than 2 rows away, then we choose CSR-Stream.
  // If this is zero (long rows) or one (final workgroup in a long row, or a single
  // row in a row block), we want to use the CSR-Vector algorithm(s).
  // We have found, through experimentation, that CSR-Vector is generally faster
  // when working on 2 rows, due to its simplicity and better reduction method.
  if (num_rows > ROWS_FOR_VECTOR)
  {
    // CSR-Stream case. See Sections III.A and III.B in the SC'14 paper:
    // Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format
    // for a detailed description of CSR-Stream.
    // In a nutshell, the idea is to use all of the threads to stream the matrix
    // values into the local memory in a fast, coalesced manner. After that, the
    // per-row reductions are done out of the local memory, which is designed
    // to handle non-coalsced accesses.

    // The best method for reducing the local memory values depends on the number
    // of rows. The SC'14 paper discusses a CSR-Scalar style reduction where
    // each thread reduces its own row. This yields good performance if there
    // are many (relatively short) rows. However, if they are few (relatively
    // long) rows, it's actually better to perform a tree-style reduction where
    // multiple threads team up to reduce the same row.

    // The calculation below tells you how many threads this workgroup can allocate
    // to each row, assuming that every row gets the same number of threads.
    // We want the closest lower (or equal) power-of-2 to this number --
    // that is how many threads can work in each row's reduction using our algorithm.
    // For instance, with workgroup size 256, 2 rows = 128 threads, 3 rows = 64
    // threads, 4 rows = 64 threads, 5 rows = 32 threads, etc.
    // int numThreadsForRed = get_local_size(0) >> ((CHAR_BIT*sizeof(unsigned
    // int))-clz(num_rows-1));
    T numThreadsForRed = wg; // Same calculation as above, done on host.

    // Stream all of this row block's matrix values into local memory.
    // Perform the matvec in parallel with this work.
    const T col = row_offset + lid;
    const T block_end = csr_row_ptr[stop_row];
    if (col + BLOCK_SIZE - WG_SIZE < block_end)
    {
      for (T i = 0; i < BLOCK_SIZE; i += WG_SIZE)
      {
        partialSums[lid + i] = alpha * csr_val[col + i] * x[csr_col_ind[col + i]];
        // alpha_mul(partialSums[lid + i], alpha, csr_val[col + i]);
        // alpha_mul(partialSums[lid + i], partialSums[lid + i], x[csr_col_ind[col + i]]);
      }
    }
    else
    {
      // This is required so that we stay in bounds for csr_val[] and csr_col_ind[].
      // Otherwise, if the matrix's endpoints don't line up with BLOCK_SIZE,
      // we will buffer overflow. On today's dGPUs, this doesn't cause problems.
      // The values are within a dGPU's page, which is zeroed out on allocation.
      // However, this may change in the future (e.g. with shared virtual memory.)
      // This causes a minor performance loss because this is the last workgroup
      // to be launched, and this loop can't be unrolled.
      for (T i = 0; col + i < csr_row_ptr[stop_row]; i += WG_SIZE)
      {
        partialSums[lid + i] = alpha * csr_val[col + i] * x[csr_col_ind[col + i]];
        // alpha_mul(partialSums[lid + i], alpha, csr_val[col + i]);
        // alpha_mul(partialSums[lid + i], partialSums[lid + i], x[csr_col_ind[col + i]]);
      }
    }
    __syncthreads();

    if (numThreadsForRed > 1)
    {
      // In this case, we want to have the workgroup perform a tree-style reduction
      // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
      // a row into {numThreadsForRed} locations in local memory.
      // After that, the entire workgroup does a parallel reduction, and each
      // row ends up with an individual answer.

      // {numThreadsForRed} adjacent threads all work on the same row, so their
      // start and end values are the same.
      // numThreadsForRed guaranteed to be a power of two, so the clz code below
      // avoids an integer divide. ~2% perf gain in EXTRA_PRECISION.
      // size_t st = lid/numThreadsForRed;
      const T local_row = row + (lid >> (31 - __clz(numThreadsForRed)));
      const T threadInBlock = lid & (numThreadsForRed - 1);

      // Not all row blocks are full -- they may have an odd number of rows. As such,
      // we need to ensure that adjacent-groups only work on real data for this rowBlock.
      if (local_row < stop_row)
      {
        const T local_first_val = csr_row_ptr[local_row] - row_offset;
        const T local_last_val = csr_row_ptr[local_row + 1] - row_offset;
        // This is dangerous -- will infinite loop if your last value is within
        // numThreadsForRed of MAX_UINT. Noticable performance gain to avoid a
        // long induction variable here, though.
        for (T local_cur_val = local_first_val + threadInBlock;
             local_cur_val < local_last_val;
             local_cur_val += numThreadsForRed)
        {
          temp_sum = temp_sum + partialSums[local_cur_val];
          // alpha_add(temp_sum, temp_sum, partialSums[local_cur_val]);
        }
      }
      __syncthreads();

      partialSums[lid] = temp_sum;

      // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
      // values sitting in the local memory. This means that, roughly, the beginning of
      // LDS is full up to {workgroup size} entries.
      // Now we perform a parallel reduction that sums together the answers for each
      // row in parallel, leaving us an answer in 'temp_sum' for each row.
      for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
      {
        __syncthreads();
        temp_sum = sum2_reduce(temp_sum, partialSums, lid, numThreadsForRed, i);
      }

      if (threadInBlock == 0 && local_row < stop_row)
      {
        // All of our write-outs check to see if the output vector should first be zeroed.
        // If so, just do a write rather than a read-write. Measured to be a slight (~5%)
        // performance improvement.
        if (beta != V{})
        {
          temp_sum = alpha_fma(beta, y[local_row], temp_sum);
        }
        // alpha_madde(temp_sum, beta, y[local_row]);
        y[local_row] = temp_sum;
      }
    }
    else
    {
      // In this case, we want to have each thread perform the reduction for a single row.
      // Essentially, this looks like performing CSR-Scalar, except it is computed out of
      // local memory.
      // However, this reduction is also much faster than CSR-Scalar, because local memory
      // is designed for scatter-gather operations.
      // We need a while loop because there may be more rows than threads in the WG.
      T local_row = row + lid;
      while (local_row < stop_row)
      {
        T local_first_val = (csr_row_ptr[local_row] - csr_row_ptr[row]);
        T local_last_val = csr_row_ptr[local_row + 1] - csr_row_ptr[row];
        temp_sum = V{};
        // alpha_setzero(temp_sum);
        for (T local_cur_val = local_first_val; local_cur_val < local_last_val;
             ++local_cur_val)
        {
          temp_sum = temp_sum + partialSums[local_cur_val];
          // alpha_add(temp_sum, temp_sum, partialSums[local_cur_val]);
        }

        // After you've done the reduction into the temp_sum register,
        // put that into the output for each row.
        if (beta != V{})
        {
          temp_sum = alpha_fma(beta, y[local_row], temp_sum);
        }
        // alpha_madde(temp_sum, beta, y[local_row]);

        y[local_row] = temp_sum;
        local_row += WG_SIZE;
      }
    }
  }
  else if (num_rows >= 1 && !wg) // CSR-Vector case.
  {
    // ^^ The above check says that if this workgroup is supposed to work on <= ROWS_VECTOR
    // number of rows then we should do the CSR-Vector algorithm. If we want this row to be
    // done with CSR-LongRows, then all of its workgroups (except the last one) will have the
    // same stop_row and row. The final workgroup in a LongRow will have stop_row and row
    // different, but the internal wg number will be non-zero.

    // If this workgroup is operating on multiple rows (because CSR-Stream is poor for small
    // numbers of rows), then it needs to iterate until it reaches the stop_row.
    // We don't check <= stop_row because of the potential for unsigned overflow.
    while (row < stop_row)
    {
      // Any workgroup only calculates, at most, BLOCK_SIZE items in this row.
      // If there are more items in this row, we use CSR-LongRows.
      temp_sum = V{};
      // alpha_setzero(temp_sum);
      const T vecStart = csr_row_ptr[row];
      const T vecEnd = csr_row_ptr[row + 1];

      // Load in a bunch of partial results into your register space, rather than LDS (no
      // contention)
      // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
      // Using a long induction variable to make sure unsigned int overflow doesn't break
      // things.
      for (T i = vecStart + lid; i < vecEnd; i += WG_SIZE)
      {
        temp_sum = alpha_fma(alpha * csr_val[i], x[csr_col_ind[i]], temp_sum);
        // ALPHA_Number tt;
        // alpha_mul(tt, alpha, csr_val[i]);
        // alpha_madde(temp_sum, tt, x[csr_col_ind[i]]);
      }

      partialSums[lid] = temp_sum;

      __syncthreads();

      // Reduce partial sums
      alpha_blockreduce_sum<WG_SIZE>(lid, partialSums);

      if (lid == 0)
      {
        temp_sum = partialSums[0];
        if (beta != V{})
        {
          temp_sum = alpha_fma(beta, y[row], temp_sum);
        }
        // alpha_madde(temp_sum, beta, y[row]);

        y[row] = temp_sum;
      }
      ++row;
    }
  }
  else
  {
    // In CSR-LongRows, we have more than one workgroup calculating this row.
    // The output values for those types of rows are stored using atomic_add, because
    // more than one parallel workgroup's value makes up the final answer.
    // Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
    // the values still left in y will be added in using the atomic_add.
    //
    // Our solution is to have the first workgroup in one of these long-rows cases
    // properly initaizlie the output vector. All the other workgroups working on this
    // row will spin-loop until that workgroup finishes its work.

    // First, figure out which workgroup you are in the row. Bottom 24 bits.
    // You can use that to find the global ID for the first workgroup calculating
    // this long row.
    T first_wg_in_row = gid - (row_blocks[gid] & ((1ULL << WG_BITS) - 1ULL));
    T compare_value = row_blocks[gid] & (1ULL << WG_BITS);

    // Bit 24 in the first workgroup is the flag that everyone waits on.
    if (gid == first_wg_in_row && lid == 0)
    {
      // The first workgroup handles the output initialization.
      U out_val = y[row];
      // V one, tt;
      V one = make_value<V>(1.f);
      // alpha_setone(one);
      temp_sum = (beta - one) * out_val;
      // alpha_sub(tt, beta, one);
      // alpha_mul(temp_sum, tt, out_val);
      atomicXor(&row_blocks[first_wg_in_row], (1ULL << WG_BITS)); // Release other workgroups.
    }
    // For every other workgroup, bit 24 holds the value they wait on.
    // If your bit 24 == first_wg's bit 24, you spin loop.
    // The first workgroup will eventually flip this bit, and you can move forward.
    __syncthreads();
    while (gid != first_wg_in_row && lid == 0 && ((atomicMax(&row_blocks[first_wg_in_row], 0ULL) & (1ULL << WG_BITS)) == compare_value))
      ;
    __syncthreads();

    // After you've passed the barrier, update your local flag to make sure that
    // the next time through, you know what to wait on.
    if (gid != first_wg_in_row && lid == 0)
      row_blocks[gid] ^= (1ULL << WG_BITS);

    // All but the final workgroup in a long-row collaboration have the same start_row
    // and stop_row. They only run for one iteration.
    // Load in a bunch of partial results into your register space, rather than LDS (no
    // contention)
    // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
    const T vecStart = alpha_mad24(
        (T)wg, (T)BLOCK_MULTIPLIER * BLOCK_SIZE, row_offset);
    const T vecEnd = min(csr_row_ptr[row + 1],
                         vecStart + BLOCK_MULTIPLIER * BLOCK_SIZE);
    for (T i = vecStart + lid; i < vecEnd; i += WG_SIZE)
    {
      // temp_sum = alpha_fma(alpha * csr_val[i], x[csr_col_ind[i]], temp_sum);
      V tt = alpha * csr_val[i];
      temp_sum = alpha_fma(tt, x[csr_col_ind[i]], temp_sum);
      // alpha_mul(tt, alpha, csr_val[i]);
      // alpha_madde(temp_sum, tt, x[csr_col_ind[i]]);
    }

    partialSums[lid] = temp_sum;

    __syncthreads();

    // Reduce partial sums
    alpha_blockreduce_sum<WG_SIZE>(lid, partialSums);

    if (lid == 0)
    {
      atomicAdd(y + row, partialSums[0]);
      // alpha_atomic_add(y[row], partialSums[0]);
    }
  }
}

/**
 * preprocess
 *
 */

template <typename T>
static inline void ComputeRowBlocks(unsigned long long *rowBlocks,
                                    size_t &rowBLOCK_SIZE,
                                    const T *rowDelimiters,
                                    T nRows,
                                    bool allocate_row_blocks = true)
{
  unsigned long long *rowBlocksBase;

  // Start at one because of rowBlock[0]
  T total_row_blocks = 1;

  if (allocate_row_blocks)
  {
    rowBlocksBase = rowBlocks;
    *rowBlocks = 0;
    ++rowBlocks;
  }

  unsigned long long sum = 0;
  unsigned long long i;
  unsigned long long last_i = 0;

  // Check to ensure nRows can fit in 32 bits
  // NOTE: There is a flaw here.
  // LCOV_EXCL_START
  if (static_cast<unsigned long long>(nRows) > static_cast<unsigned long long>(std::pow(2, ROW_BITS)))
  {
    fprintf(stderr, "nrow does not fit in 32 bits\n");
    exit(1);
  }
  // LCOV_EXCL_STOP

  T consecutive_long_rows = 0;
  for (i = 1; i <= static_cast<unsigned long long>(nRows); ++i)
  {
    T row_length = (rowDelimiters[i] - rowDelimiters[i - 1]);
    sum += row_length;

    // The following section of code calculates whether you're moving between
    // a series of "short" rows and a series of "long" rows.
    // This is because the reduction in CSR-Adaptive likes things to be
    // roughly the same length. Long rows can be reduced horizontally.
    // Short rows can be reduced one-thread-per-row. Try not to mix them.
    if (row_length > 128)
    {
      ++consecutive_long_rows;
    }
    else if (consecutive_long_rows > 0)
    {
      // If it turns out we WERE in a long-row region, cut if off now.
      if (row_length < 32) // Now we're in a short-row region
      {
        consecutive_long_rows = -1;
      }
      else
      {
        consecutive_long_rows++;
      }
    }

    // If you just entered into a "long" row from a series of short rows,
    // then we need to make sure we cut off those short rows. Put them in
    // their own workgroup.
    if (consecutive_long_rows == 1)
    {
      // Assuming there *was* a previous workgroup. If not, nothing to do here.
      if (i - last_i > 1)
      {
        if (allocate_row_blocks)
        {
          *rowBlocks = ((i - 1) << (64 - ROW_BITS));

          // If this row fits into CSR-Stream, calculate how many rows
          // can be used to do a parallel reduction.
          // Fill in the low-order bits with the numThreadsForRed
          if (((i - 1) - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
          {
            *(rowBlocks - 1) |= numThreadsForReduction<WG_SIZE>((i - 1) - last_i);
          }

          ++rowBlocks;
        }

        ++total_row_blocks;
        last_i = i - 1;
        sum = row_length;
      }
    }
    else if (consecutive_long_rows == -1)
    {
      // We see the first short row after some long ones that
      // didn't previously fill up a row block.
      if (allocate_row_blocks)
      {
        *rowBlocks = ((i - 1) << (64 - ROW_BITS));
        if (((i - 1) - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
        {
          *(rowBlocks - 1) |= numThreadsForReduction<WG_SIZE>((i - 1) - last_i);
        }

        ++rowBlocks;
      }

      ++total_row_blocks;
      last_i = i - 1;
      sum = row_length;
      consecutive_long_rows = 0;
    }

    // Now, what's up with this row? What did it do?

    // exactly one row results in non-zero elements to be greater than BLOCK_SIZE
    // This is csr-vector case; bottom WGBITS == workgroup ID
    if ((i - last_i == 1) && sum > static_cast<unsigned long long>(BLOCK_SIZE))
    {
      T numWGReq = static_cast<T>(
          std::ceil(static_cast<double>(row_length) / (BLOCK_MULTIPLIER * BLOCK_SIZE)));

      // Check to ensure #workgroups can fit in WGBITS bits, if not
      // then the last workgroup will do all the remaining work
      numWGReq = (numWGReq < static_cast<T>(std::pow(2, WG_BITS)))
                     ? numWGReq
                     : static_cast<T>(std::pow(2, WG_BITS));

      if (allocate_row_blocks)
      {
        for (T w = 1; w < numWGReq; ++w)
        {
          *rowBlocks = ((i - 1) << (64 - ROW_BITS));
          *rowBlocks |= static_cast<unsigned long long>(w);
          ++rowBlocks;
        }

        *rowBlocks = (i << (64 - ROW_BITS));
        ++rowBlocks;
      }

      total_row_blocks += numWGReq;
      last_i = i;
      sum = 0;
      consecutive_long_rows = 0;
    }
    // more than one row results in non-zero elements to be greater than BLOCK_SIZE
    // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
    else if ((i - last_i > 1) && sum > static_cast<unsigned long long>(BLOCK_SIZE))
    {
      // This row won't fit, so back off one.
      --i;

      if (allocate_row_blocks)
      {
        *rowBlocks = (i << (64 - ROW_BITS));
        if ((i - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
        {
          *(rowBlocks - 1) |= numThreadsForReduction<WG_SIZE>(i - last_i);
        }

        ++rowBlocks;
      }

      ++total_row_blocks;
      last_i = i;
      sum = 0;
      consecutive_long_rows = 0;
    }
    // This is csr-stream case; bottom WGBITS = number of parallel reduction threads
    else if (sum == static_cast<unsigned long long>(BLOCK_SIZE))
    {
      if (allocate_row_blocks)
      {
        *rowBlocks = (i << (64 - ROW_BITS));
        if ((i - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
        {
          *(rowBlocks - 1) |= numThreadsForReduction<WG_SIZE>(i - last_i);
        }

        ++rowBlocks;
      }

      ++total_row_blocks;
      last_i = i;
      sum = 0;
      consecutive_long_rows = 0;
    }
  }

  // If we didn't fill a row block with the last row, make sure we don't lose it.
  if (allocate_row_blocks && (*(rowBlocks - 1) >> (64 - ROW_BITS)) != static_cast<unsigned long long>(nRows))
  {
    *rowBlocks = (static_cast<unsigned long long>(nRows) << (64 - ROW_BITS));
    if ((nRows - last_i) > static_cast<unsigned long long>(ROWS_FOR_VECTOR))
    {
      *(rowBlocks - 1) |= numThreadsForReduction<WG_SIZE>(i - last_i);
    }

    ++rowBlocks;
  }

  ++total_row_blocks;

  if (allocate_row_blocks)
  {
    size_t dist = std::distance(rowBlocksBase, rowBlocks);
    assert((2 * dist) <= rowBLOCK_SIZE);
    // Update the size of rowBlocks to reflect the actual amount of memory used
    // We're multiplying the size by two because the extended precision form of
    // CSR-Adaptive requires more space for the final global reduction.
    rowBLOCK_SIZE = 2 * dist;
  }
  else
  {
    rowBLOCK_SIZE = 2 * total_row_blocks;
  }
}

/**
 * dispatch
 *
 */

template <typename T, typename U, typename V, typename W>
alphasparseStatus_t spmv_csr_adaptive(alphasparseHandle_t handle,
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
  cudaStream_t stream = handle->stream;

  /**
   * get row block on cpu
   */
  double time = get_time_us();

  // row blocks size
  size_t csrmv_info_size = 0;
  unsigned long long *csrmv_info_row_blocks;

  // Temporary arrays to hold device data
  std::vector<T> hptr(m + 1);
  cudaMemcpyAsync(hptr.data(), csr_row_ptr, sizeof(T) * (m + 1), cudaMemcpyDeviceToHost, stream);

  // Wait for host transfer to finish
  cudaStreamSynchronize(stream);
  // double time1 = get_time_us();
  // Determine row blocks array size
  ComputeRowBlocks((unsigned long long *)NULL, csrmv_info_size, hptr.data(), m, false);
  // double time2 = get_time_us();
  // printf("预处理1: %lf\n", (time2 - time1) / (1e3));

  // Create row blocks structure
  std::vector<unsigned long long> row_blocks(csrmv_info_size, 0);
  // double time3 = get_time_us();
  ComputeRowBlocks(row_blocks.data(), csrmv_info_size, hptr.data(), m, true);
  // double time4 = get_time_us();
  // printf("预处理2: %lf\n", (time4 - time3) / (1e3));

  // Allocate memory on device to hold csrmv info, if required
  if (csrmv_info_size > 0)
  {
    cudaMalloc((void **)&csrmv_info_row_blocks, sizeof(unsigned long long) * csrmv_info_size);

    // Copy row blocks information to device
    cudaMemcpyAsync(csrmv_info_row_blocks,
                    row_blocks.data(),
                    sizeof(unsigned long long) * csrmv_info_size,
                    cudaMemcpyHostToDevice,
                    stream);

    // Wait for device transfer to finish
    cudaStreamSynchronize(stream);

    // time = (get_time_us() - time) / (1e3);
    // printf("preprocess:%f\n", time);

    /**
     * get feature for tuning
     */
    // int32_t stream_num = 0, vector_num = 0, vectorL_num = 0;
    // int32_t block_num = (csrmv_info_size / 2) - 1;
    // for (int32_t i = 0; i < block_num; i++) {
    //     T row      = ((row_blocks[i] >> (64 - ROW_BITS)) & ((1ULL << ROW_BITS) - 1ULL));
    //     T stop_row = ((row_blocks[i + 1] >> (64 - ROW_BITS)) & ((1ULL << ROW_BITS) - 1ULL));
    //     T wg       = row_blocks[i] & ((1 << WG_BITS) - 1);
    //     T num_rows = stop_row - row;
    //     // printf("%d\n", num_rows);
    //     if (num_rows > ROWS_FOR_VECTOR)
    //         stream_num++;
    //     else if (num_rows >= 1 && !wg)
    //         vector_num++;
    //     else
    //         vectorL_num++;
    // }
    // printf("\nblocknum:%d stream:%d vector:%d vectorL:%d\n", block_num, stream_num, vector_num, vectorL_num);
  }

  /**
   * adaptive spmv on gpu
   */

  // calculate
  dim3 csrmvn_blocks((csrmv_info_size / 2) - 1);
  dim3 csrmvn_threads(WG_SIZE);
  double time5 = get_time_us();
  csrmvn_adaptive_device<<<csrmvn_blocks,
                           csrmvn_threads,
                           0,
                           stream>>>(csrmv_info_row_blocks,
                                     alpha,
                                     nnz,
                                     csr_row_ptr,
                                     csr_col_ind,
                                     csr_val,
                                     x,
                                     beta,
                                     y);
  double time6 = get_time_us();
  // printf("spmv: %lf\n", (time6 - time5) / (1e3));

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

#undef BLOCK_SIZE
#undef BLOCK_MULTIPLIER
#undef ROWS_FOR_VECTOR
#undef WG_BITS
#undef ROW_BITS
#undef WG_SIZE
