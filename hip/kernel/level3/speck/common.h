#ifndef spECK_Common_2
#define spECK_Common_2
#pragma once

/////////////// HELPERS /////////////////////////s

__host__ __device__ __forceinline__ uint32_t currentHash(uint32_t id) {
	return id * 11;
}

__device__ __forceinline__ uint32_t rowColMinMaxtoMinCol(uint32_t rowColMinMax)
{
	return rowColMinMax & ((1 << 27) - 1);
}

__device__ __forceinline__ uint32_t rowColMinMaxtoRowLength(uint32_t rowColMinMax)
{
	// printf("%lu, leads to %lu\n", (rowColMinMax >> 27), 1 << (rowColMinMax >> 27));
	return 1 << (rowColMinMax >> 27);
}

__device__ __host__ __forceinline__  uint32_t blockRangeToStartRow(uint32_t blockRange)
{
	return blockRange >> 5;
}

__device__ __host__ __forceinline__ uint32_t blockRangeToNumRows(uint32_t blockRange)
{
	return (blockRange & 0b11111) + 1;
}

template <typename INDEX_TYPE>
__device__ __forceinline__ void markRowSorted(INDEX_TYPE &column)
{
	column |= 1U << 31;
}

template <typename INDEX_TYPE>
__device__ __forceinline__ bool isRowSorted(INDEX_TYPE &column)
{
	return column & (1U << 31);
}

template <typename INDEX_TYPE>
__device__ __forceinline__ void removeSortedMark(INDEX_TYPE &column)
{
	column &= (1U << 31) - 1;
}

inline static void HandleError(hipError_t err,
							   const char *file,
							   int line)
{
	if (err != hipSuccess)
	{
		printf("%s in %s at line %d\n", hipGetErrorString(err),
			   file, line);
		throw std::exception();
	}
}

#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

template<typename T>
__host__ __device__ __forceinline__ T divup(T a, T b)
{
	return (a + b - 1) / b;
}

__device__ __forceinline__ uint32_t toRowColMinMax(uint32_t minCol, uint32_t maxCol)
{
	uint32_t width = 32U - __clz(maxCol - minCol);
	return minCol + (width << 27);
}


__device__ __host__ __forceinline__ uint32_t toBlockRange(uint32_t startRow, uint32_t numRows)
{
	return (startRow << 5) + (numRows - 1);
}

#endif