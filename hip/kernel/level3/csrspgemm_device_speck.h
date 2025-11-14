#include <hip/hip_runtime.h>
#ifndef CSRSPGEMM_DEVICE_SPECK
#define CSRSPGEMM_DEVICE_SPECK
#include "alphasparse.h"
#include <hipcub/hipcub.hpp>  
#include "speck/spECKKernels.h" 
#include "speck/spECKConfig.h" 
#include "speck/spECK_HashLoadBalancer.h" 
#include "speck/spECK_HashSpGEMM.h" 
#include "speck/HashMap.h" 
#include "speck/Config.h" 
#include "speck/limits.h" 
#include "speck/common.h" 

template <typename INDEX_TYPE, uint32_t THREADS, uint32_t rowsPerThreads>
__global__ void getLongestRowA(const INDEX_TYPE* __restrict__ rowOffsets, INDEX_TYPE* __restrict__ longestRow, const INDEX_TYPE rows, const INDEX_TYPE nnz)
{
	typedef hipcub::BlockReduce<INDEX_TYPE, THREADS> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;

	INDEX_TYPE rowLength[rowsPerThreads];

	for (int i = 0; i < rowsPerThreads; ++i)
		rowLength[i] = 0;

	INDEX_TYPE startRow = blockIdx.x * THREADS * rowsPerThreads + threadIdx.x * rowsPerThreads;
	INDEX_TYPE lastRowExcl = min(rows, blockIdx.x * THREADS * rowsPerThreads + (threadIdx.x + 1) * rowsPerThreads);

	if (lastRowExcl > startRow)
	{
		INDEX_TYPE prevOffset = rowOffsets[startRow];
		for (int i = 1; i <= lastRowExcl - startRow; ++i)
		{
			INDEX_TYPE currentRowOffset = rowOffsets[i + startRow];
			rowLength[i - 1] = currentRowOffset - prevOffset;
			prevOffset = currentRowOffset;
		}
	}

	INDEX_TYPE longestRowBlock = BlockReduce(temp_storage).Reduce(rowLength, hipcub::Max());

	if (threadIdx.x == 0)
		atomicMax(longestRow, longestRowBlock);
}

template<typename INDEX_TYPE, typename VALUE_TYPE, class T, uint32_t THREADS>
__global__ void readOperations(INDEX_TYPE A_rows, INDEX_TYPE A_nnz, INDEX_TYPE * A_row_ptr, INDEX_TYPE * A_col_idx, 
    INDEX_TYPE B_rows, INDEX_TYPE B_nnz, INDEX_TYPE * B_row_ptr, INDEX_TYPE * B_col_idx,  T *out, int rowsPerBlock, 
	INDEX_TYPE *maxComputationsPerRow, INDEX_TYPE *rowColMinMax, INDEX_TYPE *rowOperationsMax, INDEX_TYPE *sumProducts)
{
	INDEX_TYPE startRow = blockIdx.x * rowsPerBlock;
	INDEX_TYPE lastRowExcl = min(INDEX_TYPE((blockIdx.x + 1) * rowsPerBlock), INDEX_TYPE(A_rows));
	bool checkCols = rowColMinMax != nullptr;
	bool checkRowOpsMax = rowOperationsMax != nullptr;		
    
	if (startRow >= A_rows)
		return;

	__shared__ INDEX_TYPE rowOpsCounter[THREADS];
	__shared__ INDEX_TYPE rowOffsets[THREADS];
	__shared__ INDEX_TYPE rowMaxOps[THREADS];
	__shared__ INDEX_TYPE rowMinCols[THREADS];
	__shared__ INDEX_TYPE rowMaxCols[THREADS];
	__shared__ INDEX_TYPE blockProducts;
	__shared__ INDEX_TYPE blockMaxOps;

	rowOpsCounter[threadIdx.x] = 0U;
	rowMaxOps[threadIdx.x] = 0U;
	rowMinCols[threadIdx.x] = spECK::numeric_limits<INDEX_TYPE>::max();
	rowMaxCols[threadIdx.x] = 0U;
	rowOffsets[threadIdx.x] = (startRow + threadIdx.x <= lastRowExcl) ? A_row_ptr[startRow + threadIdx.x] : A_nnz;
	if (threadIdx.x == 0) {
		blockProducts = 0;
		blockMaxOps = 0;
	}

	__syncthreads();

	uint32_t startId = rowOffsets[0];
	uint32_t lastIdExcl = lastRowExcl < A_rows ? rowOffsets[rowsPerBlock] : (uint32_t)A_nnz;

	uint32_t currentRow = spECK::numeric_limits<INDEX_TYPE>::max();
	uint32_t currentRowOps = 0;
	uint32_t currentMin = spECK::numeric_limits<INDEX_TYPE>::max();
	uint32_t currentMax = 0;
	uint32_t currentRowMaxOps = 0;
	for (uint32_t id = threadIdx.x + startId; id < lastIdExcl; id += blockDim.x)
	{
		INDEX_TYPE rowA = 0;

		for(; rowA < rowsPerBlock; ++rowA)
		{
			if (rowOffsets[rowA] <= id && (rowA + startRow + 1 < A_rows ? rowOffsets[rowA + 1] : (uint32_t)A_nnz) > id)
				break;
		}

		if(currentRow != rowA)
		{
			if (currentRow != spECK::numeric_limits<INDEX_TYPE>::max()) {
				if (checkCols) {
					atomicMin(&rowMinCols[currentRow], currentMin);
					atomicMax(&rowMaxCols[currentRow], currentMax);
				}
				if(checkRowOpsMax)
					atomicMax(&rowMaxOps[currentRow], currentRowMaxOps);
				atomicAdd(&rowOpsCounter[currentRow], currentRowOps);
			}
			currentMin = spECK::numeric_limits<INDEX_TYPE>::max();
			currentMax = 0;
			currentRowMaxOps = 0;
			currentRow = rowA;
			currentRowOps = 0;
		}
		INDEX_TYPE rowB = A_col_idx[id];
		INDEX_TYPE startIdB = B_row_ptr[rowB];
		INDEX_TYPE lastIdBExcl = rowB + 1 <= B_rows ? B_row_ptr[rowB + 1] : B_nnz;
		INDEX_TYPE operations = lastIdBExcl - startIdB;

		if(checkCols && startIdB < lastIdBExcl)
		{
			currentMin = min(currentMin, B_col_idx[startIdB]);
			if (lastIdBExcl > 0)
				currentMax = max(currentMax, B_col_idx[lastIdBExcl - 1]);
		}

		currentRowOps += operations;
		if(checkRowOpsMax)
			currentRowMaxOps = max(currentRowMaxOps, operations);
	}

	if(currentRow != spECK::numeric_limits<INDEX_TYPE>::max())
	{
		if (checkCols) {
			atomicMin(&rowMinCols[currentRow], currentMin);
			atomicMax(&rowMaxCols[currentRow], currentMax);
		}
		if(checkRowOpsMax)
			atomicMax(&rowMaxOps[currentRow], currentRowMaxOps);
		atomicAdd(&rowOpsCounter[currentRow], currentRowOps);
	}

	__syncthreads();

	if (rowsPerBlock > 1) {
		INDEX_TYPE rowProducts = rowOpsCounter[threadIdx.x];
		for (int i = 16; i > 0; i /= 2)
			rowProducts += __shfl_down(rowProducts, i);

		if (threadIdx.x % 32 == 0 && rowProducts > 0)
			atomicAdd(&blockProducts, rowProducts);

		INDEX_TYPE maxRowLength = rowOpsCounter[threadIdx.x];
		for (int i = 16; i > 0; i /= 2)
			maxRowLength = max(maxRowLength, __shfl_down(maxRowLength, i));

		if (threadIdx.x % 32 == 0 && maxRowLength > 0)
			atomicMax(&blockMaxOps, maxRowLength);

		__syncthreads();
	}


	if (threadIdx.x < rowsPerBlock && (threadIdx.x + startRow) < A_rows)
	{
		out[startRow + threadIdx.x] = rowOpsCounter[threadIdx.x];
		if(checkCols)
			rowColMinMax[startRow + threadIdx.x] = toRowColMinMax(rowMinCols[threadIdx.x], rowMaxCols[threadIdx.x]);
		if(checkRowOpsMax)
			rowOperationsMax[startRow + threadIdx.x] = rowMaxOps[threadIdx.x];
	}

	if(threadIdx.x == blockDim.x - 1)
	{
		if (rowsPerBlock == 1) {
			atomicMax(maxComputationsPerRow, rowOpsCounter[0]);
			atomicAdd(sumProducts, rowOpsCounter[0]);
		}
		else {
			atomicMax(maxComputationsPerRow, blockMaxOps);
			atomicAdd(sumProducts, blockProducts);
		}
	}
}

template<typename INDEX_TYPE, typename VALUE_TYPE>
__global__ void mul_alpha(VALUE_TYPE * val, INDEX_TYPE size, const VALUE_TYPE a)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int str = blockDim.x * gridDim.x;
    for(int i = tid; i < size; i += str)
        val[i] *= a;
}

template <typename IndexType, typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
void MultiplyspECKImplementation(
                                alphasparseHandle_t handle,
                                alphasparseOperation_t opA,
                                alphasparseOperation_t opB,
                                const DataType alpha,
                                alphasparseSpMatDescr_t matA,
                                alphasparseSpMatDescr_t matB,
                                const DataType beta,
                                alphasparseSpMatDescr_t matC,
                                char * externalBuffer2)
{
    int maxStaticSharedMemoryPerBlock = handle->properties.sharedMemPerBlock;
    int maxDynamicSharedMemoryPerBlock = 1;// std::max(handle->properties.sharedMemPerBlockOptin, handle->properties.sharedMemPerBlock);
    // size_t buffer_size = 0;
    if (matB->cols > 1 << 27)
    {
        printf("ERROR: matrix B has more than %d columns (%lu)\n", 1 << 27, matB->cols);
        return;
    }
    if (matA->rows > 1 << 27)
    {
        printf("ERROR: matrix A has more than %d rows (%lu)\n", 1 << 27, matB->rows);
        return;
    }
    if (matA->nnz * matB->nnz == 0) {
        matC->nnz = 0;
        return;
    }

    if (MAX_DYNAMIC_SHARED != maxDynamicSharedMemoryPerBlock || MAX_STATIC_SHARED != maxStaticSharedMemoryPerBlock) {
        if (MAX_DYNAMIC_SHARED > maxDynamicSharedMemoryPerBlock) {
            printf("ERROR: spECK was compiled with %d maximum dynamic shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 10: spECK_DYNAMIC_MEM_PER_BLOCK\n",
                MAX_DYNAMIC_SHARED, maxDynamicSharedMemoryPerBlock);
            return;
        } else {
            printf("WARNING: spECK was compiled with %d maximum dynamic shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 10: spECK_DYNAMIC_MEM_PER_BLOCK\n",
                MAX_DYNAMIC_SHARED, maxDynamicSharedMemoryPerBlock);
        }
		if (MAX_STATIC_SHARED > MAX_DYNAMIC_SHARED)
		{
			printf("ERROR: spECK was compiled with smaller dynamic than static shared memory. (%d maximum static shared memory and %d maximum dynamic shared memory). Please check values in Multiply.h line 9 and 10",
				MAX_STATIC_SHARED, MAX_DYNAMIC_SHARED);
			return;
		}
		if (MAX_STATIC_SHARED > maxStaticSharedMemoryPerBlock)
		{
			printf("ERROR: spECK was compiled with %d maximum static shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 9: spECK_STATIC_MEM_PER_BLOCK\n",
				MAX_STATIC_SHARED, maxStaticSharedMemoryPerBlock);
			return;
		}
		else if (MAX_STATIC_SHARED < maxStaticSharedMemoryPerBlock) {
			printf("WARNING: spECK was compiled with %d maximum static shared memory, but device limit is %d. Please recompile with correct amount set in Multiply.h line 9: spECK_STATIC_MEM_PER_BLOCK\n",
				MAX_STATIC_SHARED, maxStaticSharedMemoryPerBlock);
		}
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  Constants and configs
    // -------------------------------------------------------------------------------------------------------------------------------------------

	spECKKernels spgemm(1024);

    const int kernelCountNumeric = 6;
    const int kernelCountCounting = 6;
    const int maxRowsPerBlock = 32; // this value may never exceed 32 because of some warp-optimizations
    const int warpsCounting = THREADS_PER_BLOCK / 32;
    const int warpsNumeric = THREADS_PER_BLOCK / 32;
	const int staticSharedMemPerBlockCounting = 48, staticSharedMemPerBlockNumeric = 24;
    const int sharedBytesPerWarpCounting = MAX_STATIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting; // 48 byte is the maximum static shared memory per block
    const int entriesPerWarpCounting = sharedBytesPerWarpCounting / sizeof(IndexType);
	const int sharedBytesPerBlockCounting = sharedBytesPerWarpCounting * warpsCounting;
	// CC version > 7.0 support dynamic shared memory larger than static shared
	const int dynamicSharedBytesPerWarpCounting = MAX_DYNAMIC_SHARED / warpsCounting - staticSharedMemPerBlockCounting; // 48 byte is the maximum static shared memory per block
	const int dynamicEntriesPerWarpCounting = dynamicSharedBytesPerWarpCounting / sizeof(IndexType);
	const int dynamicSharedBytesPerBlockCounting = dynamicSharedBytesPerWarpCounting * warpsCounting;

    const int sharedBytesPerWarpNumeric = MAX_STATIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
    const int entriesPerWarpNumeric = sharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
    const int sharedBytesPerBlockNumeric = sharedBytesPerWarpNumeric * warpsNumeric;
	// CC version > 7.0 support dynamic shared memory larger than static shared
    const int dynamicSharedBytesPerWarpNumeric = MAX_DYNAMIC_SHARED / warpsNumeric - staticSharedMemPerBlockNumeric; // 24 byte is the maximum static shared memory per block
	const int dynamicEntriesPerWarpNumeric = dynamicSharedBytesPerWarpNumeric / (sizeof(IndexType) + sizeof(DataType));
	const int dynamicSharedBytesPerBlockNumeric = dynamicSharedBytesPerWarpNumeric * warpsNumeric;
    assert(kernelCountCounting <= kernelCountNumeric);

    bool supportGlobalFallback = true;
    const uint32_t minimumDensityForDenseModeCounting = 999;
    const uint32_t denseModeRowThresholdInternalSorting = 999;
    const uint32_t denseModeRowThresholdExternalSorting = 18;
    const uint32_t sm = handle->properties.multiProcessorCount;
    const uint32_t cudaCores = handle->properties.multiProcessorCount * BLOCKS_PER_SM * 32;

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  INITIAL MALLOCS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    int estimatedAvgComPerRow = max(1, int((matA->nnz / matA->rows) * (matB->nnz / matB->rows)));
    // determine how many nnz of matC should be calculated by one block. avoid hashmaps running full
    int maxNnzPerBlockNumeric = entriesPerWarpNumeric * warpsNumeric * 2 / 3;
    int maxNnzPerBlockNumericDynamicSharedMem = dynamicEntriesPerWarpNumeric * warpsNumeric * 2 / 3;

    // CUDA variables
    hipStream_t stream = handle->streams[0];
    auto &streams = handle->streams;

    IndexType *blockStartRowsScale = nullptr;
    IndexType *blockCounterScale = nullptr;
    IndexType h_blockCounterScaleNumeric[kernelCountNumeric] = {0};
    IndexType h_blockCounterScaleCounting[kernelCountCounting] = {0};

    size_t cubTempBytesScan = 0;
    size_t cubTmpBytesReduce = 0;
    size_t cubTmpBytesActual = 0;
    void *cubTmp = nullptr;

    {
        hipcub::DeviceScan::ExclusiveSum(cubTmp, cubTempBytesScan, (IndexType *)matC->row_data, (IndexType *)matC->row_data, matC->rows + 1);
        hipcub::DeviceReduce::Sum(cubTmp, cubTmpBytesReduce, (IndexType *)matC->row_data, (IndexType *)matC->row_data, matC->rows);
        cubTmpBytesReduce = std::max(cubTempBytesScan, cubTmpBytesReduce);
    }

    // ----------------------------------------------------------------------------------

    uint32_t maxComputationsPerRow = 0;
    uint32_t longestRowALength = 0;

    IndexType *d_blockStartRows = nullptr;
    uint32_t *d_blockCounter = nullptr;
    uint32_t *d_rowOperations = nullptr;
    uint32_t *d_rowMaxOperations = nullptr;
    uint32_t *d_maxElementsPerRow = nullptr;
    uint32_t *d_sumProducts = nullptr;
    uint32_t *d_rowColMinMax = nullptr;
    uint32_t *d_maxComputationsPerRow = nullptr;

    uint32_t *d_combined_pointers;
    size_t d_combined_pointers_size = sizeof(uint32_t) * (4 + 2 * matA->rows) + divup(cubTempBytesScan, sizeof(uint32_t)) * sizeof(uint32_t);
    if (matA->nnz > 10000)
        d_combined_pointers_size += sizeof(uint32_t) * matA->rows;

    HANDLE_ERROR(hipMalloc(&d_combined_pointers, d_combined_pointers_size));
    d_combined_pointers = (uint32_t *)externalBuffer2;
    externalBuffer2 += d_combined_pointers_size;
    HANDLE_ERROR(hipMemsetAsync(d_combined_pointers, 0, d_combined_pointers_size));
    // buffer_size += d_combined_pointers_size;
    // printf("d_combined_pointers_size %d\n",d_combined_pointers_size);
    d_maxElementsPerRow = d_combined_pointers;
    /* keep this order */
    d_sumProducts = &d_maxElementsPerRow[1];
    d_maxComputationsPerRow = &d_sumProducts[1];
    /* until here */
    d_blockCounter = &d_maxComputationsPerRow[1];
    d_rowOperations = &d_blockCounter[1];
    d_rowMaxOperations = &d_rowOperations[matA->rows];
    cubTmp = (void *)&d_rowMaxOperations[matA->rows];
    cubTmpBytesActual = cubTempBytesScan;

    if (matA->nnz > 10000)
    {
        d_rowColMinMax = (uint32_t *)cubTmp;
        d_rowColMinMax = &d_rowColMinMax[divup(cubTempBytesScan, sizeof(uint32_t))];
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  COUNT COMPUTATIONS
    // -------------------------------------------------------------------------------------------------------------------------------------------
    uint32_t sumProducts = 0;
    // calc amount of operations per row
    {
        const uint32_t threadsPerBlock = 128U;
        // limit to threadsPerBlock rows!
        // -> and always try to stay slightly below the threads per block size, because if you are slightly above, it is way more expensive than being far below
        uint32_t rowsPerBlock = std::min(threadsPerBlock, std::max(1U, (threadsPerBlock - 8) / std::max(1U, uint32_t(matA->nnz / matA->rows))));
        rowsPerBlock = std::max(1U, std::min(rowsPerBlock, uint32_t(matA->rows) / (4U * cudaCores / threadsPerBlock)));
        readOperations<IndexType, DataType, IndexType, threadsPerBlock><<<divup(uint32_t(matA->rows), rowsPerBlock), threadsPerBlock>>>
            ((IndexType)matA->rows, (IndexType)matA->nnz, (IndexType*)matA->row_data, (IndexType*)matA->col_data, (IndexType)matB->rows, (IndexType)matB->nnz, (IndexType*)matB->row_data, (IndexType*)matB->col_data, 
            d_rowOperations, rowsPerBlock, d_maxComputationsPerRow, d_rowColMinMax, d_rowMaxOperations, d_sumProducts);

        // copying both values at once gives a huge performance boost
        uint32_t tmpArr[2];
        HANDLE_ERROR(hipMemcpy(&tmpArr, d_sumProducts, sizeof(uint32_t) * 2, hipMemcpyDeviceToHost));
        sumProducts = tmpArr[0];
        maxComputationsPerRow = tmpArr[1];
        // sumProducts = max(sumProducts, 1);
    }

    if (sumProducts == 0) {
        matC->nnz = 0;
        matC->rows = matA->rows;
        matC->cols = matB->cols;
        return;
    }

    int maxNnzPerBlockCounting = entriesPerWarpCounting * warpsCounting * 4 / 5;
    int maxNnzPerBlockCountingDynamicSharedMem = dynamicEntriesPerWarpCounting * warpsCounting * 4 / 5;

    // you always know the maximum size of the output row
    uint32_t maxRowLength = max(1, min((uint32_t)matB->cols * 12 / 10, maxComputationsPerRow));
    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  LOADBALANCE COUNTING
    // -------------------------------------------------------------------------------------------------------------------------------------------

    uint32_t h_blockCounter = 0;

    uint32_t rowsPerBlock = 1;
    if (kernelCountCounting > 5 && maxRowLength < (maxNnzPerBlockCounting >> 4)) {
        uint32_t maxRowsPerBlockUtilization = max(1, min(uint32_t(maxRowsPerBlock), uint32_t(matA->rows / (sm * BLOCKS_PER_SM << (kernelCountCounting - 2)))));
        if (maxRowLength < maxNnzPerBlockCounting >> (kernelCountCounting - 1))
        {
            if (estimatedAvgComPerRow / maxRowLength == 1 || maxRowLength / estimatedAvgComPerRow == 1)
                rowsPerBlock = min(maxRowsPerBlockUtilization, ((maxNnzPerBlockCounting >> (kernelCountCounting - 1)) / 3) / maxRowLength);
            else
                rowsPerBlock = min(maxRowsPerBlockUtilization, (maxNnzPerBlockCounting >> kernelCountCounting) / maxRowLength);
        }
        rowsPerBlock = max(rowsPerBlock, 1);
        h_blockCounterScaleCounting[kernelCountCounting - 1] = divup(uint32_t(matA->rows), rowsPerBlock);
    }
    else if (kernelCountCounting > 4 && maxRowLength < (maxNnzPerBlockCounting >> 3))
        h_blockCounterScaleCounting[4] = matA->rows;
    else if (kernelCountCounting > 3 && maxRowLength < (maxNnzPerBlockCounting >> 2))
        h_blockCounterScaleCounting[3] = matA->rows;
    else if (kernelCountCounting > 2 && maxRowLength < (maxNnzPerBlockCounting >> 1))
        h_blockCounterScaleCounting[2] = matA->rows;
    else if (kernelCountCounting > 1 && maxRowLength < (maxNnzPerBlockCounting >> 0))
        h_blockCounterScaleCounting[1] = matA->rows;
    else
        h_blockCounterScaleCounting[0] = matA->rows;
        
    uint32_t rowsRequiringGlobal = h_blockCounterScaleCounting[0];

    uint32_t actualKernelCount = min(kernelCountCounting,
                                     uint32_t(
                                         std::log2(
                                             divup(
                                                 int(maxRowLength),
                                                 min(
                                                     maxNnzPerBlockCounting >> (kernelCountCounting - 1),
                                                     maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)))) +
                                         1));

    bool useLoadBalancingCounting = false;

	// TODO check if && maxComputationsPerRow > maxNnzPerBlockCounting / 8 can be removed
    if (matA->nnz > 771843 || 
        maxComputationsPerRow < maxNnzPerBlockCountingDynamicSharedMem && maxComputationsPerRow > (maxNnzPerBlockCounting >> 2) && matA->rows > 7575 ||
        maxComputationsPerRow > maxNnzPerBlockCountingDynamicSharedMem && sumProducts > 1940177 ||
        maxComputationsPerRow / max(1, int((sumProducts / matA->rows))) > 110 && sumProducts > 1164708)
        useLoadBalancingCounting = true;

    if (useLoadBalancingCounting)
    {
        size_t combinedBlockStartSize = sizeof(IndexType) * (1 + kernelCountCounting + matA->rows * (1 + actualKernelCount));
        d_blockStartRows = (IndexType *)externalBuffer2;
        externalBuffer2 += combinedBlockStartSize;
        // printf("combinedBlockStartSize %d\n",combinedBlockStartSize);
        // HANDLE_ERROR(hipMalloc(&d_blockStartRows, combinedBlockStartSize));
        blockStartRowsScale = &d_blockStartRows[matA->rows + 1];
        blockCounterScale = &blockStartRowsScale[actualKernelCount * matA->rows];
        HANDLE_ERROR(hipMemset(blockCounterScale, 0, sizeof(IndexType) * kernelCountCounting));
        // buffer_size += combinedBlockStartSize;
        // load balance over amount of operations per row in A
        spgemm.h_AssignHashSpGEMMBlocksToRowsOfSameSizeOperations<uint32_t, DataType, uint8_t, kernelCountCounting>(
            matA, matB, d_rowOperations, blockStartRowsScale, blockCounterScale, h_blockCounterScaleCounting, d_blockStartRows,
            maxNnzPerBlockCounting, maxNnzPerBlockCountingDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
    }
    else
    {
        h_blockCounter = matA->rows;
        d_blockStartRows = nullptr;
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  ALLOCATE GLOBAL MAPS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    int elementsPerMap = (std::max(maxRowLength, uint32_t(maxNnzPerBlockCountingDynamicSharedMem)) * 5) / 4;
    supportGlobalFallback &= maxRowLength > entriesPerWarpCounting * warpsCounting;

    typedef HashMap<uint32_t, DataType> GlobalMap;
    typedef HashMapNoValue<uint32_t, 1> GlobalMapRowOffsets;
    typedef HashMapNoValue<uint32_t, maxRowsPerBlock> GlobalMapNoValue;
    void *hashMaps = nullptr;
    IndexType *maps_indices = nullptr;
    DataType *maps_values = nullptr;
    uint32_t hashMapCount = 0;
    size_t globalMapMaxSize;
    globalMapMaxSize = std::max(sizeof(GlobalMap), sizeof(GlobalMapNoValue));
    globalMapMaxSize = std::max(globalMapMaxSize, sizeof(GlobalMapRowOffsets));

    if (supportGlobalFallback)
    {
        hashMapCount = std::min((IndexType)(sm * BLOCKS_PER_SM), h_blockCounterScaleCounting[0]);
        hashMapCount = std::min(hashMapCount, rowsRequiringGlobal);
        supportGlobalFallback &= hashMapCount > 0;
    }

    rowsRequiringGlobal = matB->cols < entriesPerWarpCounting * warpsCounting ? 0 : rowsRequiringGlobal;
    bool isDenseCounting = useLoadBalancingCounting && rowsRequiringGlobal > 0 && maxComputationsPerRow > maxNnzPerBlockCountingDynamicSharedMem * 2;

    if (isDenseCounting)
    {
        supportGlobalFallback = false;
        // every bit is one column
        if (matB->cols > (warpsCounting * sharedBytesPerWarpCounting * 8) / 2)
        {
            if (longestRowALength == 0)
            {
                uint32_t *d_longestRowALength = nullptr;
                d_longestRowALength = (uint32_t *)externalBuffer2;
                externalBuffer2 += sizeof(uint32_t);
                // HANDLE_ERROR(hipMalloc(&d_longestRowALength, sizeof(uint32_t)));
                HANDLE_ERROR(hipMemset(d_longestRowALength, 0, sizeof(uint32_t)));
                // buffer_size += sizeof(uint32_t);
                const uint32_t blockdim = 256;
                const uint32_t rowsPerThread = 2;
                const uint32_t blocks = divup(uint32_t(matA->rows), blockdim * rowsPerThread);
                getLongestRowA<IndexType, blockdim, rowsPerThread><<<blocks, blockdim>>>((IndexType*)matA->row_data, d_longestRowALength, (IndexType)matA->rows, (IndexType)matA->nnz);
                hipMemcpy(&longestRowALength, d_longestRowALength, sizeof(uint32_t), hipMemcpyDeviceToHost);
            }
            
            // only use global maps if the row cursors can't be held in shared memory
            if (elementsPerMap * 2 > warpsCounting * entriesPerWarpCounting)
            {
                hashMapCount = sm * BLOCKS_PER_SM;
                elementsPerMap = longestRowALength * 5 / 4;

                if (maps_indices != nullptr)
                    HANDLE_ERROR(hipFree(maps_indices));
                if (hashMaps != nullptr)
                    HANDLE_ERROR(hipFree(hashMaps));

                HANDLE_ERROR(hipMalloc(&maps_indices, sizeof(uint32_t) * hashMapCount * (elementsPerMap + maxRowsPerBlock + 1)));
                HANDLE_ERROR(hipMalloc(&hashMaps, globalMapMaxSize * hashMapCount));

                spgemm.setLaunchDimensions(hashMapCount, streams[0], 32 * warpsNumeric);
                spgemm.h_InitializeGlobalMapsNoVal<GlobalMapRowOffsets, uint32_t>((GlobalMapRowOffsets *)hashMaps, hashMapCount, maps_indices, elementsPerMap, maxRowsPerBlock);
            }
        }
    }
    // printf("buffer_size %d\n", buffer_size);
    if (supportGlobalFallback)
    {
        HANDLE_ERROR(hipMalloc(&hashMaps, globalMapMaxSize * hashMapCount));
        HANDLE_ERROR(hipMalloc(&maps_indices, sizeof(IndexType) * hashMapCount * (elementsPerMap + maxRowsPerBlock + 1)));

        spgemm.setLaunchDimensions(hashMapCount, streams[0], 32 * warpsCounting);
        spgemm.h_InitializeGlobalMapsNoVal<GlobalMapNoValue, IndexType>((GlobalMapNoValue *)hashMaps, hashMapCount, maps_indices, elementsPerMap, maxRowsPerBlock);
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  PRE-COUNTING LOAD-OPTIMIZATION
    // -------------------------------------------------------------------------------------------------------------------------------------------

    IndexType blockPrefixScaled[kernelCountCounting] = {0};
    {
        uint32_t activeSM = h_blockCounterScaleCounting[0];
        // never go up to top level
        int firstXEmpty = h_blockCounterScaleCounting[0] == 0;
        bool foundFirstNonEmpty = h_blockCounterScaleCounting[0] != 0;
        for (int i = 1; i < kernelCountCounting; ++i)
        {
            blockPrefixScaled[i] = h_blockCounterScaleCounting[i - 1] + blockPrefixScaled[i - 1];
            activeSM += 2 * h_blockCounterScaleCounting[i] >> (i - 1);
            if (!foundFirstNonEmpty)
            {
                if (h_blockCounterScaleCounting[i] == 0)
                    firstXEmpty++;
                else
                    foundFirstNonEmpty = true;
            }
        }

        // avoid div by zero
        activeSM = max(activeSM, 1);

        if (activeSM < sm * BLOCKS_PER_SM)
        {
            int shiftUp = min(firstXEmpty, int(std::log2(sm * BLOCKS_PER_SM / activeSM)));

            if (shiftUp > 0)
            {
                for (int i = 0; i < kernelCountCounting; i++)
                {
                    if (i + shiftUp < kernelCountCounting)
                    {
                        h_blockCounterScaleCounting[i] = h_blockCounterScaleCounting[i + shiftUp];
                        blockPrefixScaled[i] = blockPrefixScaled[i + shiftUp];
                    }
                    else
                    {
                        h_blockCounterScaleCounting[i] = 0;
                        blockPrefixScaled[i] = h_blockCounter;
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  COUNT NNZ PER ROW OF C
    // -------------------------------------------------------------------------------------------------------------------------------------------
    {   
        if (h_blockCounterScaleCounting[0] > 0)
        {
            if (isDenseCounting)
            {
                // this only uses 1 block per sm and therefore hash 50% occupancy, but better caching
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[0], streams[0], (32 * warpsCounting >> 0), dynamicSharedBytesPerBlockCounting);
                spgemm.h_DenseSpGEMMCount<IndexType, DataType, GlobalMapRowOffsets, dynamicSharedBytesPerBlockCounting, true, (32 * warpsCounting >> 0)>(
                    matA, matB, (GlobalMapRowOffsets *)hashMaps, hashMapCount, (IndexType *)matC->row_data, d_blockStartRows + blockPrefixScaled[0],
                    d_rowOperations, h_blockCounterScaleCounting[0], d_rowColMinMax,
                    d_rowMaxOperations, d_maxElementsPerRow, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleCounting[0], streams[0], 32 * warpsCounting >> 0, dynamicSharedBytesPerBlockCounting);
                spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, dynamicSharedBytesPerBlockCounting, true, (32 * warpsCounting >> 0)>(
                    matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, (IndexType *)matC->row_data, d_rowOperations,
                    d_blockStartRows + blockPrefixScaled[0], h_blockCounterScaleCounting[0], d_rowColMinMax,
                    d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
            }
        }
    
        if (kernelCountCounting > 1 && h_blockCounterScaleCounting[1] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[1], streams[1], 32 * warpsCounting >> 0, sharedBytesPerBlockCounting >> 0);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 0), false, (32 * warpsCounting >> 0)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, (IndexType *)matC->row_data, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[1], h_blockCounterScaleCounting[1], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 2 && h_blockCounterScaleCounting[2] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[2], streams[2], (32 * warpsCounting >> 1), sharedBytesPerBlockCounting >> 1);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 1), false, (32 * warpsCounting >> 1)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, (IndexType *)matC->row_data, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[2], h_blockCounterScaleCounting[2], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 3 && h_blockCounterScaleCounting[3] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[3], streams[3], (32 * warpsCounting >> 2), sharedBytesPerBlockCounting >> 2);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 2), false, (32 * warpsCounting >> 2)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, (IndexType *)matC->row_data, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[3], h_blockCounterScaleCounting[3], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 4 && h_blockCounterScaleCounting[4] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[4], streams[4], 32 * warpsCounting >> 3, sharedBytesPerBlockCounting >> 3);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 3), false, (32 * warpsCounting >> 3)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, (IndexType *)matC->row_data, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[4], h_blockCounterScaleCounting[4], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }

        if (kernelCountCounting > 5 && h_blockCounterScaleCounting[5] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleCounting[5], streams[5], 32 * warpsCounting >> 4, sharedBytesPerBlockCounting >> 4);
            spgemm.h_SpGEMMCountLauncher<IndexType, DataType, maxRowsPerBlock, GlobalMapNoValue, GlobalMapRowOffsets, (sharedBytesPerBlockCounting >> 4), false, (32 * warpsCounting >> 4)>(
                matA, matB, (GlobalMapNoValue *)hashMaps, hashMapCount, nullptr, 0, (IndexType *)matC->row_data, d_rowOperations,
                d_blockStartRows + blockPrefixScaled[5], h_blockCounterScaleCounting[5], d_rowColMinMax,
                d_rowMaxOperations, minimumDensityForDenseModeCounting, d_maxElementsPerRow, rowsPerBlock);
        }     
    }    
    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  SCAN ROW OFFSETS AND GET NNZ OF C
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // now we need to allocate that memory for prefix scan and for finding the longest row
    if (cubTmpBytesActual < cubTempBytesScan)
    {
        cubTmpBytesActual = cubTempBytesScan;
        if (cubTmp != nullptr)
            HANDLE_ERROR(hipFree(cubTmp));
        HANDLE_ERROR(hipMalloc(&cubTmp, cubTmpBytesActual));
    }
    IndexType tnnz;
    // prefix sum to get the starting ids of each row of mat C
    hipcub::DeviceScan::ExclusiveSum(cubTmp, cubTmpBytesActual, (IndexType *)matC->row_data, (IndexType *)matC->row_data, matC->rows + 1);
    {
        IndexType nnz;
        HANDLE_ERROR(hipMemcpy(&nnz, (IndexType *)matC->row_data + matC->rows, sizeof(IndexType), hipMemcpyDeviceToHost));
        tnnz = nnz;
    }
    printf("nnz %d\n",tnnz);
    if(tnnz != matC->nnz)
    {
        matC->nnz = tnnz;
        if(matC->col_data != nullptr)
            hipFree(matC->col_data);
        if(matC->val_data != nullptr)
            hipFree(matC->val_data);
        HANDLE_ERROR(hipMalloc((void **)&matC->col_data, sizeof(IndexType)*matC->nnz));
        HANDLE_ERROR(hipMalloc((void **)&matC->val_data, sizeof(DataType)*matC->nnz));
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  LOAD BALANCE NUMERIC
    // -------------------------------------------------------------------------------------------------------------------------------------------

    uint32_t maxElementsPerRow = maxRowLength;
    HANDLE_ERROR(hipMemcpy(&maxElementsPerRow, d_maxElementsPerRow, sizeof(uint32_t), hipMemcpyDeviceToHost));
    
    bool reprocessLoadBalanceNumeric = useLoadBalancingCounting;
    rowsPerBlock = 1;
    
    // get the longest row in order to minimize the global map size which needs to be allocated

    if (kernelCountNumeric > 5 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 4)) {
        uint32_t maxRowsPerBlockUtilization = max(1, min(uint32_t(maxRowsPerBlock), uint32_t(matA->rows / (sm * BLOCKS_PER_SM << (kernelCountNumeric - 2)))));
        if (maxElementsPerRow<(entriesPerWarpNumeric * warpsNumeric)>> kernelCountNumeric)
        {
            if (maxElementsPerRow / max(1U, uint32_t(matC->nnz / matC->rows)) == 1)
                rowsPerBlock = min(maxRowsPerBlockUtilization, (maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)) / maxElementsPerRow);
            else
                rowsPerBlock = min(maxRowsPerBlockUtilization, (entriesPerWarpNumeric * warpsNumeric >> (kernelCountNumeric - 1)) / maxElementsPerRow);
        }
        rowsPerBlock = max(rowsPerBlock, 1);
        h_blockCounterScaleNumeric[kernelCountNumeric - 1] = divup(uint32_t(matA->rows), rowsPerBlock);
    }
    else if (kernelCountNumeric > 4 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 3))
        h_blockCounterScaleNumeric[4] = matC->rows;
    else if (kernelCountNumeric > 3 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 2))
        h_blockCounterScaleNumeric[3] = matC->rows;
    else if (kernelCountNumeric > 2 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 1))
        h_blockCounterScaleNumeric[2] = matC->rows;
    else if (kernelCountNumeric > 1 && maxElementsPerRow < (maxNnzPerBlockNumeric >> 0))
        h_blockCounterScaleNumeric[1] = matC->rows;
    else
        h_blockCounterScaleNumeric[0] = matC->rows;

    supportGlobalFallback = true;
    supportGlobalFallback &= maxElementsPerRow >= maxNnzPerBlockNumericDynamicSharedMem;
    rowsRequiringGlobal = h_blockCounterScaleNumeric[0];
    
    uint32_t avgElementsPerRow = max(1, int(matC->nnz / matC->rows));
    uint32_t maxAvgElementsPerRowRatio = maxElementsPerRow / avgElementsPerRow;
    reprocessLoadBalanceNumeric = false;
    if (maxElementsPerRow > (maxNnzPerBlockNumeric >> 2) && matA->rows >= 1236 && sumProducts > 636293 ||
        maxElementsPerRow > (maxNnzPerBlockNumeric >> (kernelCountNumeric - 1)) && (
            maxAvgElementsPerRowRatio > 4 && sumProducts > 4921876 ||
            maxAvgElementsPerRowRatio > 13 && sumProducts > 385847 ||
            maxAvgElementsPerRowRatio > 18 && sumProducts > 26263 && avgElementsPerRow > 22 ||
            maxAvgElementsPerRowRatio > 146))
        reprocessLoadBalanceNumeric = true;

    // can bring a performance benefit for some matrices, but has small overhead
    if (reprocessLoadBalanceNumeric && matC->nnz > 0)
    {
        if (d_blockCounter == nullptr)
        {
            HANDLE_ERROR(hipMalloc(&d_blockCounter, sizeof(uint32_t)));
        }
        if (blockCounterScale == nullptr)
        {
            // size_t combinedBlockStartSize = sizeof(IndexType) * (1 + kernelCountNumeric + matA->rows * (1 + actualKernelCount));

            // HANDLE_ERROR(hipMalloc(&d_blockStartRows, combinedBlockStartSize));
            blockStartRowsScale = &d_blockStartRows[matA->rows + 1];
            blockCounterScale = &blockStartRowsScale[actualKernelCount * matA->rows];
        }
        // reset buffers
        HANDLE_ERROR(hipMemsetAsync(d_blockCounter, 0, sizeof(uint32_t)));
        HANDLE_ERROR(hipMemsetAsync(blockCounterScale, 0, sizeof(IndexType) * kernelCountNumeric));
        // printf("maxNnzPerBlockNumeric %d maxNnzPerBlockNumericDynamicSharedMem %d maxRowsPerBlock %d actualKernelCount %d rowsRequiringGlobal %d\n", maxNnzPerBlockNumeric, maxNnzPerBlockNumericDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
        spgemm.h_AssignHashSpGEMMBlocksToRowsOfSameSize<IndexType, DataType, uint8_t, kernelCountNumeric>(
            matC, (IndexType *)matC->row_data, blockStartRowsScale, d_blockStartRows, blockCounterScale, h_blockCounterScaleNumeric,
            maxNnzPerBlockNumeric, maxNnzPerBlockNumericDynamicSharedMem, maxRowsPerBlock, actualKernelCount, rowsRequiringGlobal);
    }
    else
    {
        // HANDLE_ERROR(hipFree(d_blockStartRows));
        d_blockStartRows = nullptr;
    }
    // for(int i = 0; i < kernelCountNumeric; i++) printf("h_blockCounterScaleNumeric[%d]= %d %d\n", i, h_blockCounterScaleNumeric[i], matC->row_data != nullptr); 
    // if(matC->row_data == nullptr)
    //     HANDLE_ERROR(hipMalloc((void **)&matC->row_data, sizeof(IndexType)*(matC->rows + 1)));

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  ALLOCATE GLOBAL MAPS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // always disabled since we always use dense mode for large rows
    supportGlobalFallback = false;
    if (supportGlobalFallback)
    {
        // update elements per map now that we know the lengths of each row --> could save some global memory and therefore allocation time
        elementsPerMap = max(maxElementsPerRow, maxNnzPerBlockNumericDynamicSharedMem) * 3 / 2;
        supportGlobalFallback &= h_blockCounterScaleNumeric[0] > 0;
        hashMapCount = min(sm * BLOCKS_PER_SM, h_blockCounterScaleNumeric[0]);
        hashMapCount = min(hashMapCount, rowsRequiringGlobal);
        supportGlobalFallback &= hashMapCount > 0;
    }

    rowsRequiringGlobal = matB->cols < entriesPerWarpNumeric * warpsNumeric ? 0 : rowsRequiringGlobal;
    bool isDenseOutput = h_blockCounterScaleNumeric[0] > 0;

    GlobalMapRowOffsets *rowOffsetMaps = nullptr;
    IndexType *rowOffsetMapIndices = nullptr;
    uint32_t rowOffsetMapCount = 0;
    uint32_t rowOffsetMapElementsPer = 0;

    if (isDenseOutput)
    {
        if (longestRowALength == 0)
        {
            uint32_t *d_longestRowALength = nullptr;
            HANDLE_ERROR(hipMalloc(&d_longestRowALength, sizeof(uint32_t)));
            HANDLE_ERROR(hipMemset(d_longestRowALength, 0, sizeof(uint32_t)));

            const uint32_t _threads = 256;
            const uint32_t rowsPerThread = 2;
            const uint32_t blocks = divup(uint32_t(matA->rows), _threads * rowsPerThread);
            getLongestRowA<IndexType, _threads, rowsPerThread><<<blocks, _threads>>>((IndexType*)matA->row_data, d_longestRowALength, (IndexType)matA->rows, (IndexType)matA->nnz);

            hipMemcpy(&longestRowALength, d_longestRowALength, sizeof(uint32_t), hipMemcpyDeviceToHost);
        }
        // printf("longestRowALength %d\n", longestRowALength);
        rowOffsetMapElementsPer = longestRowALength;
        rowOffsetMapCount = min(h_blockCounterScaleNumeric[0], sm * BLOCKS_PER_SM);

        // only allocate global maps if row cursors can't be held in share memory
        if (elementsPerMap * 2 * sizeof(IndexType) > warpsNumeric * entriesPerWarpNumeric * (sizeof(IndexType) + sizeof(DataType)))
        {
            if (h_blockCounterScaleNumeric[0] != 0)
            {
                if (rowOffsetMaps != nullptr)
                    HANDLE_ERROR(hipFree(rowOffsetMaps));
                HANDLE_ERROR(hipMalloc(&rowOffsetMaps, globalMapMaxSize * rowOffsetMapCount));

                if (rowOffsetMapIndices != nullptr)
                {
                    HANDLE_ERROR(hipFree(rowOffsetMapIndices));
                    rowOffsetMapIndices = nullptr;
                }

                if (rowOffsetMapIndices == nullptr)
                    HANDLE_ERROR(hipMalloc(&rowOffsetMapIndices, sizeof(IndexType) * rowOffsetMapCount * (rowOffsetMapElementsPer + maxRowsPerBlock + 1)));

                spgemm.setLaunchDimensions(rowOffsetMapCount, stream, 32 * warpsNumeric);
                spgemm.h_InitializeGlobalMapsNoVal<GlobalMapRowOffsets, uint32_t>((GlobalMapRowOffsets *)rowOffsetMaps, rowOffsetMapCount, rowOffsetMapIndices, rowOffsetMapElementsPer, maxRowsPerBlock);
            }
        }
    }

    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  PRE-NUMERIC LOAD OPTIMIZATIONS
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // alloc indices for rows which shall be sorted by cub
    bool sortAllInplace = false;
    {
        {

            uint32_t activeSM = h_blockCounterScaleNumeric[0];
            // never go up to top level
            int firstXEmpty = 0;
            bool foundFirstNonEmpty = h_blockCounterScaleNumeric[0] != 0;
            for (int i = 1; i < kernelCountNumeric; ++i)
            {
                blockPrefixScaled[i] = h_blockCounterScaleNumeric[i - 1] + blockPrefixScaled[i - 1];
                activeSM += 2 * h_blockCounterScaleNumeric[i] >> (i - 1);
                if (!foundFirstNonEmpty)
                {
                    if (h_blockCounterScaleNumeric[i] == 0)
                        firstXEmpty++;
                    else
                        foundFirstNonEmpty = true;
                }
            }

            // avoid div by zero
            activeSM = max(activeSM, 1);

            if (activeSM < sm * BLOCKS_PER_SM)
            {
                int shiftUp = min(firstXEmpty, int(std::log2(sm * BLOCKS_PER_SM / activeSM)));

                if (shiftUp > 0)
                {
                    if (firstXEmpty >= 2)
                        sortAllInplace = true;

                    for (int i = 0; i < kernelCountNumeric; i++)
                    {
                        if (i + shiftUp < kernelCountNumeric)
                        {
                            h_blockCounterScaleNumeric[i] = h_blockCounterScaleNumeric[i + shiftUp];
                            blockPrefixScaled[i] = blockPrefixScaled[i + shiftUp];
                        }
                        else
                        {
                            h_blockCounterScaleNumeric[i] = 0;
                            blockPrefixScaled[i] = h_blockCounter;
                        }
                    }
                }
            }
        }

        // inplace starts to be faster if the size of the maps is getting smaller
		Config::SortModes sortMode = Config::SortModes::CubSegmentedSort;

        const uint32_t entrySize = sizeof(IndexType) + sizeof(DataType);

        Config::SpGEMMMethods spGemmMethodNumeric = Config::AutoSpGEMM;
        // Config::SpGEMMMethods spGemmMethodNumeric = Config::DenseSpGEMM;

        // -------------------------------------------------------------------------------------------------------------------------------------------
        //  NUMERIC SPGEMM
        // -------------------------------------------------------------------------------------------------------------------------------------------
        if (h_blockCounterScaleNumeric[0] > 0)
        {   
            spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[0], streams[0], 32 * warpsNumeric, dynamicSharedBytesPerBlockNumeric);
            spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, dynamicSharedBytesPerBlockNumeric, false, (32 * warpsNumeric)>(
                matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMapRowOffsets *)rowOffsetMaps, rowOffsetMapCount,
                d_blockStartRows, d_rowOperations, h_blockCounterScaleNumeric[0], d_rowColMinMax,
                d_rowMaxOperations, false, rowsPerBlock);
        }
        // if (h_blockCounterScaleNumeric[0] > 0)
        // {
        //     spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[0], streams[0], 32 * warpsNumeric, sharedBytesPerBlockNumeric);
        //     spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, sharedBytesPerBlockNumeric, true, (32 * warpsNumeric)>(
        //         matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
        //         d_blockStartRows, d_rowOperations,
        //         Config::InPlace,
        //         h_blockCounterScaleNumeric[0], d_rowColMinMax, 
        //         d_rowMaxOperations, denseModeRowThresholdExternalSorting, false, rowsPerBlock);
        // }

        sortMode = sortAllInplace ? Config::InPlace : Config::Separate;
        // HANDLE_ERROR(hipDeviceSynchronize());
        // IndexType * tmp = (IndexType*)malloc(sizeof(IndexType)*(50));
        // HANDLE_ERROR(hipMemcpy(tmp, (IndexType *)matC->col_data, sizeof(IndexType)*(50), hipMemcpyDeviceToHost));
        // for(int i = 0; i < 50; i++) printf("tmp[%d]=%d\n",i,tmp[i]);
        bool setSortingBit = sortAllInplace ? false : maxElementsPerRow >= 500;
        if (kernelCountNumeric > 1 && h_blockCounterScaleNumeric[1] > 0)
        {   
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {   
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], (32 * warpsNumeric >> 0), (sharedBytesPerBlockNumeric >> 0));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric >> 0)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[1], d_rowColMinMax, 
                    d_rowMaxOperations, denseModeRowThresholdExternalSorting, setSortingBit, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric >> 0, (sharedBytesPerBlockNumeric >> 0));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric >> 0)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                    h_blockCounterScaleNumeric[1], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric, (sharedBytesPerBlockNumeric >> 0));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 0), false, (32 * warpsNumeric)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[1], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[1], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
        }
   
        if (kernelCountNumeric > 2 && h_blockCounterScaleNumeric[2] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {   
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], (32 * warpsNumeric >> 1), (sharedBytesPerBlockNumeric >> 1));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[2], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdExternalSorting, setSortingBit, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric >> 1, (sharedBytesPerBlockNumeric >> 1));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                    h_blockCounterScaleNumeric[2], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric >> 1, (sharedBytesPerBlockNumeric >> 1));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 1), false, (32 * warpsNumeric >> 1)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[2], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[2], d_rowColMinMax,
                    d_rowMaxOperations, setSortingBit, rowsPerBlock);
            }
        }

        sortMode = Config::InPlace;

        if (kernelCountNumeric > 3 && h_blockCounterScaleNumeric[3] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {   
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], (32 * warpsNumeric >> 2), (sharedBytesPerBlockNumeric >> 2));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[3], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], 32 * warpsNumeric >> 2, (sharedBytesPerBlockNumeric >> 2));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                    h_blockCounterScaleNumeric[3], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[3], streams[3], 32 * warpsNumeric >> 2, (sharedBytesPerBlockNumeric >> 2));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 2), false, (32 * warpsNumeric >> 2)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[3], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[3], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
        }

        if (kernelCountNumeric > 4 && h_blockCounterScaleNumeric[4] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM)
            {   
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], (32 * warpsNumeric >> 3), (sharedBytesPerBlockNumeric >> 3));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[4], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], 32 * warpsNumeric >> 3, (sharedBytesPerBlockNumeric >> 3));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                    h_blockCounterScaleNumeric[4], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
            else
            {
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[4], streams[4], 32 * warpsNumeric >> 3, (sharedBytesPerBlockNumeric >> 3));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 3), false, (32 * warpsNumeric >> 3)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[4], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[4], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
        }

        if (kernelCountNumeric > 5 && h_blockCounterScaleNumeric[5] > 0)
        {
            if (spGemmMethodNumeric == Config::AutoSpGEMM || ((rowsPerBlock > 1 || reprocessLoadBalanceNumeric) && spGemmMethodNumeric != Config::HashSpGEMM))
            {   
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], (32 * warpsNumeric >> 4), (sharedBytesPerBlockNumeric >> 4));
                spgemm.h_SpGEMMNumericLauncher<IndexType, DataType, GlobalMap, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount, rowOffsetMaps, rowOffsetMapCount,
                    d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[5], d_rowColMinMax,
                    d_rowMaxOperations, denseModeRowThresholdInternalSorting, false, rowsPerBlock);
            }
            else if (spGemmMethodNumeric == Config::DenseSpGEMM)
            {
                
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], 32 * warpsNumeric >> 4, (sharedBytesPerBlockNumeric >> 4));
                spgemm.h_DenseSpGEMMNumeric<IndexType, DataType, GlobalMapRowOffsets, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMapRowOffsets *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                    h_blockCounterScaleNumeric[5], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
            else
            {   
                spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[5], streams[5], 32 * warpsNumeric >> 4, (sharedBytesPerBlockNumeric >> 4));
                spgemm.h_HashSpGEMMNumeric<IndexType, DataType, GlobalMap, (sharedBytesPerBlockNumeric >> 4), false, (32 * warpsNumeric >> 4)>(
                    matA, matB, (IndexType *)matC->row_data, (IndexType *)matC->col_data, (DataType *)matC->val_data, alpha, matC, (GlobalMap *)hashMaps, hashMapCount,
                    d_blockStartRows + blockPrefixScaled[5], d_rowOperations,
                    sortMode,
                    h_blockCounterScaleNumeric[5], d_rowColMinMax,
                    d_rowMaxOperations, false, rowsPerBlock);
            }
        }
    }
    
    // mul_alpha<IndexType, DataType><<<1024, 256>>>((DataType *)matC->val_data,(IndexType)matC->nnz,alpha);
    // HANDLE_ERROR(hipDeviceSynchronize());
    // DataType * val = (DataType *)malloc(matC->nnz*sizeof(DataType));
    // HANDLE_ERROR(hipMemcpy(val, (DataType *)matC->val_data, matC->nnz*sizeof(DataType), hipMemcpyDeviceToHost));
    // for(int i = 0; i < 50; i++) printf("i %d = %f\n", i, val[i]);
    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  SORT MEDIUM AND LONG ROWS
    // -------------------------------------------------------------------------------------------------------------------------------------------
    // HANDLE_ERROR(hipDeviceSynchronize());
    if (!sortAllInplace && (h_blockCounterScaleNumeric[1] + h_blockCounterScaleNumeric[2] > 0) && maxElementsPerRow >= 500)
    {
        if (h_blockCounterScaleNumeric[2] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[2], streams[2], 32 * warpsNumeric / 4);
            spgemm.h_HashSpGEMMSorting<uint32_t, DataType, 32 * warpsNumeric / 4, entriesPerWarpNumeric * 32 / 2>(
                matC, d_blockStartRows + blockPrefixScaled[2], h_blockCounterScaleNumeric[2], true);
        }

        if (h_blockCounterScaleNumeric[1] > 0)
        {
            spgemm.setLaunchDimensions(h_blockCounterScaleNumeric[1], streams[1], 32 * warpsNumeric / 2);
            spgemm.h_HashSpGEMMSorting<uint32_t, DataType, 32 * warpsNumeric / 2, entriesPerWarpNumeric * 32>(
                matC, d_blockStartRows + blockPrefixScaled[1], h_blockCounterScaleNumeric[1], true);
        }
    }
    // HANDLE_ERROR(hipDeviceSynchronize());
    // -------------------------------------------------------------------------------------------------------------------------------------------
    //  FREE ALLOCATED MEMORY
    // -------------------------------------------------------------------------------------------------------------------------------------------

    // if (d_blockStartRows != nullptr)
    //     HANDLE_ERROR(hipFree(d_blockStartRows));
    if (hashMaps != nullptr)
        HANDLE_ERROR(hipFree(hashMaps));
    if (maps_indices != nullptr)
        HANDLE_ERROR(hipFree(maps_indices));
    if (maps_values != nullptr)
        HANDLE_ERROR(hipFree(maps_values));

    // if (d_combined_pointers != nullptr)
    //     HANDLE_ERROR(hipFree(d_combined_pointers));

    if (rowOffsetMaps != nullptr)
        HANDLE_ERROR(hipFree(rowOffsetMaps));
    if (rowOffsetMapIndices != nullptr)
        HANDLE_ERROR(hipFree(rowOffsetMapIndices));

     // -------------------------------------------------------------------------------------------------------------------------------------------
    //  END
    // -------------------------------------------------------------------------------------------------------------------------------------------
    // DataType * val = (DataType *)malloc(matC->nnz*sizeof(DataType));
    // HANDLE_ERROR(hipMemcpy(val, matC->val_data, matC->nnz*sizeof(DataType), hipMemcpyDeviceToHost));
    // for(int i = 0; i < 50; i++) printf("i %d = %f\n", i, val[i]);
    // matOut.rows = matC->rows;
    // matOut.cols = matC->cols;
    // matOut.nnz = matC->nnz;
    // matOut.col_ids = matC.col_ids;
    // matOut.row_offsets = matC->row_data;
    // matOut.data = matC.data;
}
#endif