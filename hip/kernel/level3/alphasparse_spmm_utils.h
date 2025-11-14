#pragma once

#define RefThreadPerBlock 256
#define PHYSICAL_WARP_SIZE 64

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a < b) ? b : a)

template<typename T, 
         unsigned int warp_size, 
         unsigned int subwarp_size>
__device__ __forceinline__ T warpReduce(T sum) {
  if (warp_size >= 64 && subwarp_size < 64) sum += __shfl_down(sum, 32);
  if (warp_size >= 32 && subwarp_size < 32) sum += __shfl_down(sum, 16);
  if (warp_size >= 16 && subwarp_size < 16) sum += __shfl_down(sum, 8);
  if (warp_size >= 8 && subwarp_size < 8) sum += __shfl_down(sum, 4);
  if (warp_size >= 4 && subwarp_size < 4) sum += __shfl_down(sum, 2);
  if (warp_size >= 2 && subwarp_size < 2) sum += __shfl_down(sum, 1);
  return sum;
}

template<typename T>
static __device__ __forceinline__
T binary_search_segment_number(const T *segoffsets, const T n_seg,
                                 const T elem_id) {
  T lo = 1, hi = n_seg, mid;
  while (lo <= hi) {
    mid = (lo + hi) >> 1;
    if (segoffsets[mid] <= elem_id) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return hi;
}

template<typename T, typename V, typename W>
static __global__ void
matrix_scalar_multiply_kernel(T M, T N, W alpha, V *matA) {
  T tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < M * N) {
    matA[tid] = alpha * matA[tid];
  }
}