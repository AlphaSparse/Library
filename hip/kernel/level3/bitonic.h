#include <hip/hip_runtime.h>
const int SHARED_SIZE_INT = 2048;
const int THREADS_PER_BLOCK = 64;
const int WARP_SIZE = 64;



template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_SharedMemBucketSize_sync(
    int *sharedMem, const int sharedMemSize, const bool increasing);

__device__ void bitonic_sort_on_fast( int *sharedMem, const int sharedMemSize )
{
  CreateBitonicSequence_SharedMemBucketSize_sync<256 , 512>( sharedMem , sharedMemSize , true );
}

// Launch params for all kernels.
template <int SHARED_SIZE_INT_PARAM, int THREADS_PER_BLOCK_PARAM>
struct KernelParams
{
public:
  enum
  {
    SHARED_SIZE_INT = SHARED_SIZE_INT_PARAM,
    THREADS_PER_BLOCK = THREADS_PER_BLOCK_PARAM,
    WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE,
    WARP_BUCKET_SIZE = 512,
    COMPARE_SWAP_WARP_BUCKET_SIZE = WARP_BUCKET_SIZE / 2
  };
};
// template <int SHARED_SIZE_INT_PARAM, int THREADS_PER_BLOCK_PARAM>
// struct KernelParams
//{
// public:
//   enum
//   {
//     SHARED_SIZE_INT = 2048,
//     THREADS_PER_BLOCK = 64,
//     WARPS_PER_BLOCK = 1,
//     WARP_BUCKET_SIZE = 512,
//     COMPARE_SWAP_WARP_BUCKET_SIZE = WARP_BUCKET_SIZE / 2
//   };
// };

////////////////////////////////////////////////////////////////////////////////////
bool BitonicSort(int *in_out_gpuDataPtr, int dataSize);

template <typename KERNEL_PARAMS>
__global__ void SortBitionicInSharedMemoryKernel(int *data, int dataSize_int);

template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void SortBitionicInSharedMemory(
    int *globalDataPtrForThisBlock, int *sharedMem,
    const int sharedMemSize, const bool increasing);

__device__ void MemcpyData_sync(
    const int *__restrict__ srcMemPtr, int *__restrict__ dstMemPtr,
    const int memSize);


template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicSort_short_sync(int *warpSharedMemPtr, int sharedMemSize);

template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_SharedMemBucketSize_sync(
    int *sharedMem, const int sharedMemSize, const bool increasing);

template <int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_WarpBucketSize_sync(
    int *sharedMem, const int sharedMemSize);

template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicSort_BucketSizeWide_sync(int *warpSharedMemPtr);

template <bool SORT_INCREASING>
__device__ int PerformBitonicSort_32ElementsWide_sync(int val);

__device__ __forceinline__ int SelectMinMax(const int val1, const int val2,
                                            bool takeMax)

{
  // Returns min or max of val1 and val2 depeding on takeMax param.
  // if(takeMax) return max(val1, val2);
  // else return min(val1, val2);
  return ((val1 > val2) ^ (!takeMax)) ? val1 : val2;
}
template <unsigned long int THREAD_MASK, int STEP_OFFSET>
__device__ __forceinline__ int PerformStep_32ElementsWide_sync(const int val)
{
  // offset进行数据交换，thread_mask确定哪些线程取最大
  const int otherVal = __shfl_xor( val, STEP_OFFSET, 64);
      // __shfl_xor(val, STEP_OFFSET, 64);
  // const unsigned long int thisThreadMaskInWarp = 1 << threadIdx.x;
  // const bool shouldTakeMax = (thisThreadMaskInWarp & THREAD_MASK);
  const bool shouldTakeMax = (THREAD_MASK >> threadIdx.x & 1);
  const int output = SelectMinMax(val, otherVal, shouldTakeMax);

  return output;
}

template <bool SORT_INCREASING>
__device__ int PerformStage_WarpWide_sync(int val);

template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformStageForGivenBucketSize_sync(int *threadRegisterBuffer);

template <typename T, bool PRODUCE_INCREASING>
__device__ bool NeedsToBeSwapped(const T &val1, const T &val2)
{
  return PRODUCE_INCREASING ? (val1 > val2) : (val1 < val2);
}

template <int BUCKET_SIZE, int COMPARE_SWAP_WARP_BUCKET_SIZE,
          int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageDynamicOrdering_BucketSize_sync(
    int *sharedMem, const int sharedMemSize);

template <bool PRODUCE_INCREASING, int INITIAL_BUCKET_SIZE,
          int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageFixedOrdering_BucketSize_sync(
    int *sharedMem, const int sharedMemSize);

template <int WARP_BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void CompareSwap_sync(int *sharedMem, const int bucketOffset);

template <int INITIAL_BUCKET_SIZE, int WARP_BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicStage_WarpBucketSize_sync(
    int *sharedMem, const int sharedMemSize);

template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformStage_BucketSizeWide_sync(int *warpSharedMemPtr);

template <typename T>
__device__ void Swap(T &val1, T &val2)
{
  T temp = val1;
  val1 = val2;
  val2 = temp;
}

////////////////////////////////////////////////////////////////////////////////////
bool BitonicSort(int *in_out_gpuDataPtr, int dataSize)
{
  if ((dataSize & (dataSize - 1)) != 0)
    return false; // Not power of two.

  using PARAMS = KernelParams<SHARED_SIZE_INT, THREADS_PER_BLOCK>;
  int *in_out_gpuDataPtr_int = in_out_gpuDataPtr;
  const int dataSize_int = dataSize;
  const int numOfBlocks = dataSize_int / PARAMS::THREADS_PER_BLOCK;

  // Sorts chunks of shared mem size.
  SortBitionicInSharedMemoryKernel<PARAMS><<<1, THREADS_PER_BLOCK>>>(
      in_out_gpuDataPtr_int, dataSize_int);

  return true;
}
////////////////////////////////////////////////////////////////////////////////////
template <typename KERNEL_PARAMS>
__global__ void SortBitionicInSharedMemoryKernel(int *data, int dataSize)
{
  __shared__ int sharedMem[KERNEL_PARAMS::SHARED_SIZE_INT];

  SortBitionicInSharedMemory<
      KERNEL_PARAMS::COMPARE_SWAP_WARP_BUCKET_SIZE,
      KERNEL_PARAMS::WARP_BUCKET_SIZE>(data, sharedMem,
                                       dataSize,
                                       1);
}
////////////////////////////////////////////////////////////////////////////
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void SortBitionicInSharedMemory(
    int *globalDataPtrForThisBlock, int *sharedMem,
    const int sharedMemSize, const bool increasing)
{
  MemcpyData_sync(globalDataPtrForThisBlock, sharedMem,
                  sharedMemSize);

  CreateBitonicSequence_SharedMemBucketSize_sync<
      COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
      sharedMem, sharedMemSize, increasing);

  MemcpyData_sync(sharedMem, globalDataPtrForThisBlock,
                  sharedMemSize);
}
////////////////////////////////////////////////////////////////////////////
__device__ void MemcpyData_sync(
    const int *__restrict__ srcMemPtr, int *__restrict__ dstMemPtr,
    const int memSize)
{
  for (int allThreadsStride = 0; allThreadsStride < memSize;
       allThreadsStride += blockDim.x)
    dstMemPtr[allThreadsStride + threadIdx.x] =
        srcMemPtr[allThreadsStride + threadIdx.x];

  __syncthreads();
}
////////////////////////////////////////////////////////////////////////////
template <int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_SharedMemBucketSize_sync(
    int *sharedMem, const int sharedMemSize, const bool increasing)
{

  if (sharedMemSize < 512)
    PerformBitonicSort_short_sync<WARP_BUCKET_SIZE, true>(sharedMem, sharedMemSize);
  else
    CreateBitonicSequence_WarpBucketSize_sync<WARP_BUCKET_SIZE>(sharedMem, sharedMemSize);

  if (sharedMemSize >= 1024)
    PerformBitonicStageDynamicOrdering_BucketSize_sync<
        1024, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(sharedMem,
                                                               sharedMemSize);
  if (sharedMemSize >= 2048)
    PerformBitonicStageDynamicOrdering_BucketSize_sync<
        2048, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(sharedMem,
                                                               sharedMemSize);
}

///////////////////////////////////////////////////////////////////////////////////
template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicSort_short_sync(int *warpSharedMemPtr, int sharedMemSize)
{

  const int itemsPerThread = BUCKET_SIZE / WARP_SIZE; // 8
  int storage[itemsPerThread];                        // 每个线程处理连续的itemsPerThread个

  // Load data into registers.
  storage[0] = warpSharedMemPtr[threadIdx.x + 0 * WARP_SIZE];
  if (sharedMemSize > 64)
    storage[1] = warpSharedMemPtr[threadIdx.x + 1 * WARP_SIZE];
  if (sharedMemSize > 128)
  {
    storage[2] = warpSharedMemPtr[threadIdx.x + 2 * WARP_SIZE];
    storage[3] = warpSharedMemPtr[threadIdx.x + 3 * WARP_SIZE];
  }
  if (sharedMemSize > 256)
  {
    storage[4] = warpSharedMemPtr[threadIdx.x + 4 * WARP_SIZE];
    storage[5] = warpSharedMemPtr[threadIdx.x + 5 * WARP_SIZE];
    storage[6] = warpSharedMemPtr[threadIdx.x + 6 * WARP_SIZE];
    storage[7] = warpSharedMemPtr[threadIdx.x + 7 * WARP_SIZE];
  }

  // Stages 1 - 5. warp bucket构建双调序列
  {
    storage[0] = PerformBitonicSort_32ElementsWide_sync<true>(storage[0]);
    if (sharedMemSize > 64)
      storage[1] = PerformBitonicSort_32ElementsWide_sync<false>(storage[1]);
    if (sharedMemSize > 128)
    {
      storage[2] = PerformBitonicSort_32ElementsWide_sync<true>(storage[2]);
      storage[3] = PerformBitonicSort_32ElementsWide_sync<false>(storage[3]);
    }
    if (sharedMemSize > 256)
    {
      storage[4] = PerformBitonicSort_32ElementsWide_sync<true>(storage[4]);
      storage[5] = PerformBitonicSort_32ElementsWide_sync<false>(storage[5]);
      storage[6] = PerformBitonicSort_32ElementsWide_sync<true>(storage[6]);
      storage[7] = PerformBitonicSort_32ElementsWide_sync<false>(storage[7]);
    }
  }

  // Stage 6.
  {
    if (sharedMemSize > 64)
      PerformStageForGivenBucketSize_sync<128, true>(storage + 0);
    if (sharedMemSize > 128)
      PerformStageForGivenBucketSize_sync<128, false>(storage + 0 + 2);

    if (sharedMemSize > 256)
    {
      PerformStageForGivenBucketSize_sync<128, true>(storage + 4);
      PerformStageForGivenBucketSize_sync<128, false>(storage + 4 + 2);
    }
  }

  // Stage 7.
  if (sharedMemSize > 128)
    PerformStageForGivenBucketSize_sync<256, true>(storage);
  if (sharedMemSize > 256)
    PerformStageForGivenBucketSize_sync<256, false>(storage + 4);

  // Stage 8.
  if (sharedMemSize > 256)
    PerformStageForGivenBucketSize_sync<512, PRODUCE_INCREASING>(storage);

  // Store date to sharedMem.
  warpSharedMemPtr[threadIdx.x + 0 * WARP_SIZE] = storage[0];
  if (sharedMemSize > 64)
    warpSharedMemPtr[threadIdx.x + 1 * WARP_SIZE] = storage[1];
  if (sharedMemSize > 128)
  {
    warpSharedMemPtr[threadIdx.x + 2 * WARP_SIZE] = storage[2];
    warpSharedMemPtr[threadIdx.x + 3 * WARP_SIZE] = storage[3];
  }
  if (sharedMemSize > 256)
  {
    warpSharedMemPtr[threadIdx.x + 4 * WARP_SIZE] = storage[4];
    warpSharedMemPtr[threadIdx.x + 5 * WARP_SIZE] = storage[5];
    warpSharedMemPtr[threadIdx.x + 6 * WARP_SIZE] = storage[6];
    warpSharedMemPtr[threadIdx.x + 7 * WARP_SIZE] = storage[7];
  }

  // __syncthreads();
}

////////////////////////////////////////////////////////////////////////////
template <int WARP_BUCKET_SIZE>
__device__ void CreateBitonicSequence_WarpBucketSize_sync(
    int *sharedMem, const int sharedMemSize)
{
  int thisWarpSharedMemOffset = 0;

  for (thisWarpSharedMemOffset = 0; thisWarpSharedMemOffset + WARP_BUCKET_SIZE < sharedMemSize; thisWarpSharedMemOffset += 2 * WARP_BUCKET_SIZE)
  {
    PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, true>(sharedMem + thisWarpSharedMemOffset);
    PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, false>(sharedMem + thisWarpSharedMemOffset + WARP_BUCKET_SIZE);
  }
  if( sharedMemSize%(WARP_BUCKET_SIZE*2) != 0   )
     PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, true>(sharedMem + thisWarpSharedMemOffset);
  // if( ( sharedMemSize%(2*WARP_BUCKET_SIZE) )!=0 )
  //   PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, true>(sharedMem + thisWarpSharedMemOffset);

  // 512*1
  // PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, true>(sharedMem + thisWarpSharedMemOffset * 0);
  // __syncthreads();
  // 512*2
  // PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, false>(sharedMem + thisWarpSharedMemOffset * 1);
  // __syncthreads();
  // 512*3
  // PerformBitonicSort_BucketSizeWide_sync<WARP_BUCKET_SIZE, true>(sharedMem + thisWarpSharedMemOffset * 2);
  // __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////////
template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicSort_BucketSizeWide_sync(int *warpSharedMemPtr)
{

  const int itemsPerThread = BUCKET_SIZE / WARP_SIZE; // 8
  int storage[itemsPerThread];                        // 每个线程处理连续的itemsPerThread个

  // Load data into registers.
#pragma unroll
  for (int i = 0; i < itemsPerThread; ++i)
    storage[i] = warpSharedMemPtr[threadIdx.x + i * WARP_SIZE];

    // Stages 1 - 5. warp bucket构建双调序列
#pragma unroll
  for (int i = 0; i < itemsPerThread; i += 2)
  {
    storage[i] = PerformBitonicSort_32ElementsWide_sync<true>(
        storage[i]);
    storage[i + 1] =
        PerformBitonicSort_32ElementsWide_sync<false>(
            storage[i + 1]);
  }

  // Stage 6.
#pragma unroll
  for (int i = 0; i < itemsPerThread; i += 4)
  {
    PerformStageForGivenBucketSize_sync<128, true>(storage + i);
    PerformStageForGivenBucketSize_sync<128, false>(storage + i +
                                                    2);
  }

  // Stage 7.
  PerformStageForGivenBucketSize_sync<256, true>(storage);
  PerformStageForGivenBucketSize_sync<256, false>(storage + 4);

  // Stage 8.
  PerformStageForGivenBucketSize_sync<512, PRODUCE_INCREASING>(storage);

  // Store date to sharedMem.
#pragma unroll
  for (int i = 0; i < itemsPerThread; ++i)
    warpSharedMemPtr[threadIdx.x + i * WARP_SIZE] =
        storage[i];

  // __syncthreads();
}
////////////////////////////////////////////////////////////////////////////
// 构建左降右升的双调序列
template <bool SORT_INCREASING>
__device__ int PerformBitonicSort_32ElementsWide_sync(int val)
{
  // Stage 1 0110 0110 0110 0110 0110 0110 0110 0110
  val = PerformStep_32ElementsWide_sync<0x6666666666666666, 1>(val); // 以2*1为组互换 0-1

  // Stage 2 0011 1100 0011 1100 0011 1100 0011 1100
  val = PerformStep_32ElementsWide_sync<0x3C3C3C3C3C3C3C3C, 2>(val); // 以2*2为组互换 0 1 2 3 - 2 3 0 1
  //         0101 1010 0101 1010 0101 1010 0101 1010
  val = PerformStep_32ElementsWide_sync<0x5A5A5A5A5A5A5A5A, 1>(val);

  // Stage 3 0000 1111 1111 0000 0000 1111 1111 0000
  val = PerformStep_32ElementsWide_sync<0x0FF00FF00FF00FF0, 4>(val); // 以2*4为组互换 0 1 2 3 4 5 6 7 - 4 5 6 7 0 1 2 3
  //         0011 0011 1100 1100 0011 0011 1100 1100
  val = PerformStep_32ElementsWide_sync<0x33CC33CC33CC33CC, 2>(val);
  //         0101 0101 1010 1010 0101 0101 1010 1010
  val = PerformStep_32ElementsWide_sync<0x55AA55AA55AA55AA, 1>(val);

  // Stage 4 0000 0000 1111 1111 1111 1111 0000 0000
  val = PerformStep_32ElementsWide_sync<0x00FFFF0000FFFF00, 8>(val);
  //         0000 1111 0000 1111 1111 0000 1111 0000
  val = PerformStep_32ElementsWide_sync<0x0F0FF0F00F0FF0F0, 4>(val);
  //         0011 0011 0011 0011 1100 1100 1100 1100
  val = PerformStep_32ElementsWide_sync<0x3333CCCC3333CCCC, 2>(val);
  //         0101 0101 0101 0101 1010 1010 1010 1010
  val = PerformStep_32ElementsWide_sync<0x5555AAAA5555AAAA, 1>(val);

  // Stage 5 0000 0000 0000 0000 1111 1111 1111 1111
  val = PerformStep_32ElementsWide_sync<0x0000FFFFFFFF0000, 16>(val);
  //         0000 0000 1111 1111 0000 0000 1111 1111
  val = PerformStep_32ElementsWide_sync<0x00FF00FFFF00FF00, 8>(val);
  //         0000 1111 0000 1111 0000 1111 0000 1111
  val = PerformStep_32ElementsWide_sync<0x0F0F0F0FF0F0F0F0, 4>(val);
  //         0011 0011 0011 0011 0011 0011 0011 0011
  val = PerformStep_32ElementsWide_sync<0x33333333CCCCCCCC, 2>(val);
  //         0101 0101 0101 0101 0101 0101 0101 0101
  val = PerformStep_32ElementsWide_sync<0x55555555AAAAAAAA, 1>(val);

  // Stage 6
  return PerformStage_WarpWide_sync<SORT_INCREASING>(val);
}
////////////////////////////////////////////////////////////////////////////
// // offset进行数据交换，thread_mask确定哪些线程取最大
// template <int THREAD_MASK, int STEP_OFFSET>
// __device__ int PerformStep_32ElementsWide_sync(const int val)
// {
//   const int otherVal =
//       __shfl_xor(0xFFFFFFFFFFFFFFFF, val, STEP_OFFSET);
//   const int thisThreadMaskInWarp = 1 << threadIdx.x();
//   const bool shouldTakeMax = (thisThreadMaskInWarp & THREAD_MASK);
//   const int output = helpers::SelectMinMax(val, otherVal, shouldTakeMax);

//   return output;
// }
// ////////////////////////////////////////////////////////////////////////////
// // Returns min or max of val1 and val2 depeding on takeMax param.
// // if(takeMax) return max(val1, val2);
// // else return min(val1, val2);
// __device__ __forceinline__ int SelectMinMax(const int val1, const int val2,
//                                             bool takeMax)

// {
//   // Branchless version commented out - it is slower on Turing:
//   // --------------------------------------------
//   // const bool comp = (val1 > val2) ^ (!takeMax);
//   // return val2 ^ ((val1 ^ val2) & -(comp));
//   // -------------------------------------------

//   // MNMX version:
//   // NOTE: I would like to generate just one MNMX instruction with predicate
//   // cond, however I don't know how to force compiler to do it since ptx only
//   // have separate min and max functions. Following code generates similar SASS:
//   // @!P0 IMNMX R17,R14,R16, PT
//   // @P0  IMNMX R17,R14,R16, !PT
//   // what I need:
//   //      IMNMX R17,R14,R16, P0
//   // -----------------------------------------------
//   // if ( takeMax )
//   //   return SelectMinMax<true>(val1,val2);
//   // else
//   //   return SelectMinMax<false>(val1,val2);
//   // -----------------------------------------------

//   return ((val1 > val2) ^ (!takeMax)) ? val1 : val2;
// }
////////////////////////////////////////////////////////////////////////////
template <bool SORT_INCREASING>
__device__ int PerformStage_WarpWide_sync(int val)
{
  if (SORT_INCREASING)
  {
    // 1111 1111 1111 1111 1111 1111 1111 1111
    val = PerformStep_32ElementsWide_sync<0xFFFFFFFF00000000, 32>(val);
    // 1111 1111 1111 1111 0000 0000 0000 0000
    val = PerformStep_32ElementsWide_sync<0xFFFF0000FFFF0000, 16>(val);
    // 1111 1111 0000 0000 1111 1111 0000 0000
    val = PerformStep_32ElementsWide_sync<0xFF00FF00FF00FF00, 8>(val);
    // 1111 0000 1111 0000 1111 0000 1111 0000
    val = PerformStep_32ElementsWide_sync<0xF0F0F0F0F0F0F0F0, 4>(val);
    // 1100 1100 1100 1100 1100 1100 1100 1100
    val = PerformStep_32ElementsWide_sync<0xCCCCCCCCCCCCCCCC, 2>(val);
    // 1010 1010 1010 1010 1010 1010 1010 1010
    val = PerformStep_32ElementsWide_sync<0xAAAAAAAAAAAAAAAA, 1>(val);
  }
  else // sort decreasing
  {
    // 0000 0000 0000 0000 0000 0000 0000 0000
    val = PerformStep_32ElementsWide_sync<0x00000000FFFFFFFF, 32>(val);
    // 0000 0000 0000 0000 1111 1111 1111 1111
    val = PerformStep_32ElementsWide_sync<0x0000FFFF0000FFFF, 16>(val);
    // 0000 0000 1111 1111 0000 0000 1111 1111
    val = PerformStep_32ElementsWide_sync<0x00FF00FF00FF00FF, 8>(val);
    // 0000 1111 0000 1111 0000 1111 0000 1111
    val = PerformStep_32ElementsWide_sync<0x0F0F0F0F0F0F0F0F, 4>(val);
    // 0011 0011 0011 0011 0011 0011 0011 0011
    val = PerformStep_32ElementsWide_sync<0x3333333333333333, 2>(val);
    // 0101 0101 0101 0101 0101 0101 0101 0101
    val = PerformStep_32ElementsWide_sync<0x5555555555555555, 1>(val);
  }
  return val;
}

////////////////////////////////////////////////////////////////////////////
template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformStageForGivenBucketSize_sync(int *threadRegisterBuffer)
{
  const int REGISTER_BUFFER_SIZE = BUCKET_SIZE / WARP_SIZE;
#pragma unroll
  for (int offset = BUCKET_SIZE / 2; offset >= WARP_SIZE; offset /= 2)
  {
    const int storageOffset = offset / WARP_SIZE;

#pragma unroll
    for (int storageIdx = 0; storageIdx < REGISTER_BUFFER_SIZE; ++storageIdx)
    {
      const int compareToIdx = storageIdx ^ storageOffset;

      // NOTE: do nothing if compareToIdx < storageIdx. This means that the loop
      // does something only for REGISTER_BUFFER_SIZE/2 items!
      if (compareToIdx < storageIdx)
        continue;

      if (NeedsToBeSwapped<int, PRODUCE_INCREASING>(
              threadRegisterBuffer[storageIdx],
              threadRegisterBuffer[compareToIdx]))
        Swap(threadRegisterBuffer[storageIdx],
             threadRegisterBuffer[compareToIdx]);
    }
  }

#pragma unroll
  for (int i = 0; i < REGISTER_BUFFER_SIZE; ++i)
  {
    threadRegisterBuffer[i] =
        PerformStage_WarpWide_sync<PRODUCE_INCREASING>(
            threadRegisterBuffer[i]);
  }
}
////////////////////////////////////////////////////////////////////////////
// template <typename T, bool PRODUCE_INCREASING>
// __device__ bool NeedsToBeSwapped(const T &val1, const T &val2)
// {
//   return PRODUCE_INCREASING ? (val1 > val2) : (val1 < val2);
// }
// ////////////////////////////////////////////////////////////////////////////
// template <typename T>
// __device__ void Swap(T &val1, T &val2)
// {
//   T temp = val1;
//   val1 = val2;
//   val2 = temp;
// }
////////////////////////////////////////////////////////////////////////////
template <int BUCKET_SIZE, int COMPARE_SWAP_WARP_BUCKET_SIZE,
          int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageDynamicOrdering_BucketSize_sync(
    int *sharedMem, const int sharedMemSize)
{
  int thisBucketOffset = 0;
#pragma unroll
  for (thisBucketOffset = 0; thisBucketOffset + BUCKET_SIZE < sharedMemSize; thisBucketOffset += 2 * BUCKET_SIZE)
  {
    PerformBitonicStageFixedOrdering_BucketSize_sync<
        true, BUCKET_SIZE, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem + thisBucketOffset, sharedMemSize);
    PerformBitonicStageFixedOrdering_BucketSize_sync<
        false, BUCKET_SIZE, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem + thisBucketOffset + BUCKET_SIZE, sharedMemSize);
  }
  if ((sharedMemSize % (2 * BUCKET_SIZE)) != 0)
    PerformBitonicStageFixedOrdering_BucketSize_sync<
        true, BUCKET_SIZE, COMPARE_SWAP_WARP_BUCKET_SIZE, WARP_BUCKET_SIZE>(
        sharedMem + thisBucketOffset, sharedMemSize);
}
////////////////////////////////////////////////////////////////////////////
template <bool PRODUCE_INCREASING, int INITIAL_BUCKET_SIZE,
          int COMPARE_SWAP_WARP_BUCKET_SIZE, int WARP_BUCKET_SIZE>
__device__ void PerformBitonicStageFixedOrdering_BucketSize_sync(
    int *sharedMem, const int sharedMemSize)
{
#pragma unroll
  for (int currentBucketSize = INITIAL_BUCKET_SIZE;
       currentBucketSize > WARP_BUCKET_SIZE; currentBucketSize /= 2)
  {
    const int currentBucketStride = currentBucketSize / 2;

#pragma unroll
    for (int thisWarpSharedMemOffset0 = 0;
         thisWarpSharedMemOffset0 < INITIAL_BUCKET_SIZE;
         thisWarpSharedMemOffset0 += currentBucketSize)
    {
#pragma unroll
      for (int thisWarpSharedMemOffset1 = 0;
           thisWarpSharedMemOffset1 < currentBucketStride;
           thisWarpSharedMemOffset1 += COMPARE_SWAP_WARP_BUCKET_SIZE)
        CompareSwap_sync<COMPARE_SWAP_WARP_BUCKET_SIZE, PRODUCE_INCREASING>(
            sharedMem + thisWarpSharedMemOffset0 + thisWarpSharedMemOffset1, currentBucketStride);
    }
    // __syncthreads();
  }

  PerformBitonicStage_WarpBucketSize_sync<INITIAL_BUCKET_SIZE,
                                          WARP_BUCKET_SIZE,
                                          PRODUCE_INCREASING>(
      sharedMem, sharedMemSize);
}
///////////////////////////////////////////////////////////////////////////////////
template <int WARP_BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void CompareSwap_sync(int *sharedMem, const int bucketOffset)
{
  const int ITEMS_PER_THREAD = WARP_BUCKET_SIZE / WARP_SIZE;
#pragma unroll
  // 为了降低bank conflict，依然每个线程处理8个元素，但是线程之间任务连续分布
  for (int i = 0; i < ITEMS_PER_THREAD; ++i)
  {
    const int offset1 = i * WARP_SIZE + threadIdx.x;
    const int offset2 =
        bucketOffset + i * WARP_SIZE + threadIdx.x;
    const int val1 = sharedMem[offset1];
    const int val2 = sharedMem[offset2];

    const bool swapNeeded =
        NeedsToBeSwapped<int, PRODUCE_INCREASING>(val1, val2);
    if (swapNeeded)
    {
      sharedMem[offset1] = val2;
      sharedMem[offset2] = val1;
    }
  }
}
////////////////////////////////////////////////////////////////////////////
template <int INITIAL_BUCKET_SIZE, int WARP_BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformBitonicStage_WarpBucketSize_sync(
    int *sharedMem, const int sharedMemSize)
{

  for (int thisWarpSharedMemOffset = 0;
       thisWarpSharedMemOffset < INITIAL_BUCKET_SIZE;
       thisWarpSharedMemOffset += WARP_BUCKET_SIZE)
  {
    // 同样，此循环也是考虑到超长shared memory情形，当前配置下无需循环
    PerformStage_BucketSizeWide_sync<WARP_BUCKET_SIZE,
                                     PRODUCE_INCREASING>(
        sharedMem + thisWarpSharedMemOffset);
  }
  // __syncthreads();
}
///////////////////////////////////////////////////////////////////////////////////
template <int BUCKET_SIZE, bool PRODUCE_INCREASING>
__device__ void PerformStage_BucketSizeWide_sync(int *warpSharedMemPtr)
{
  const int itemsPerThread = BUCKET_SIZE / WARP_SIZE;
  int storage[itemsPerThread];

// Load data into registers.
#pragma unroll
  for (int i = 0; i < itemsPerThread; ++i)
    storage[i] = warpSharedMemPtr[threadIdx.x + i * WARP_SIZE];

  // Sort in registers.
  PerformStageForGivenBucketSize_sync<BUCKET_SIZE,
                                      PRODUCE_INCREASING>(
      storage);

// Store sorted sequence to sharedMem.
#pragma unroll
  for (int i = 0; i < itemsPerThread; ++i)
    warpSharedMemPtr[threadIdx.x + i * WARP_SIZE] = storage[i];

  // __syncthreads();
}
