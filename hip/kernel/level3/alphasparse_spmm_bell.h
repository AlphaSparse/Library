#include "hip/hip_runtime.h"
#include "alphasparse.h"

template<int BELL_BLOCK_DIM,
         int BLK_SIZE_Y,
         typename T,
         typename U,
         typename V,
         typename W>
__global__ void
bellmm_general_blockdim_device(alphasparseOperation_t trans_A,
                               alphasparseOperation_t trans_B,
                               alphasparseOrder_t order_B,
                               alphasparseOrder_t order_C,
                               alphasparseDirection_t dir_A,
                               T Mb,
                               T N,
                               W alpha,
                               T bell_cols,
                               T block_dim,
                               const T* __restrict__ bell_col_ind,
                               const U* __restrict__ bell_val,
                               const U* __restrict__ B,
                               T ldb,
                               W beta,
                               V* __restrict__ C,
                               T ldc,
                               alphasparseIndexBase_t idx_base)
{
  const T tidx = threadIdx.x;
  const T tidy = threadIdx.y;
  const T block_row = blockIdx.x;
  const T bell_width = (block_row < Mb) ? (bell_cols / block_dim) : T{};

  __shared__ U shared_B[BELL_BLOCK_DIM * BLK_SIZE_Y];
  __shared__ U shared_A[BELL_BLOCK_DIM * BELL_BLOCK_DIM];

  const T global_col = tidy + blockIdx.y * BLK_SIZE_Y;
  const T colB = global_col * ldb;

  for (T x = 0; x < block_dim; x += BELL_BLOCK_DIM) {
    const T global_row = tidx + x + blockIdx.x * block_dim;

    U sum = U{};

    for (T j = 0; j < bell_width; ++j) {
      const T ell_idx = j * Mb + block_row;
      const T block_col = (bell_col_ind[ell_idx] - idx_base);

      for (T y = 0; y < block_dim; y += BLK_SIZE_Y) {
        const bool is_A_valid =
          ((tidx + x) < block_dim && (tidy + y) < block_dim) &&
          (block_col >= 0);
        const bool is_B_valid =
          ((global_col < N) && ((tidx + y) < block_dim)) && (block_col >= 0);

        if ((trans_B == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             order_B == ALPHASPARSE_ORDER_COL) ||
            (trans_B != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             order_B != ALPHASPARSE_ORDER_COL)) {
          shared_B[BELL_BLOCK_DIM * tidy + tidx] =
            (is_B_valid) ? B[block_dim * block_col + (tidx + y) + colB] : U{};
        } else {
          shared_B[BELL_BLOCK_DIM * tidy + tidx] =
            (is_B_valid)
              ? B[global_col + ldb * (block_dim * block_col + (tidx + y))]
              : U{};
        }
        if (dir_A == ALPHASPARSE_DIRECTION_ROW) {
          shared_A[BELL_BLOCK_DIM * tidy + tidx] =
            (is_A_valid) ? bell_val[block_dim * block_dim * ell_idx +
                                    block_dim * (tidx + x) + (tidy + y)]
                         : U{};
        } else {
          shared_A[BELL_BLOCK_DIM * tidy + tidx] =
            (is_A_valid) ? bell_val[block_dim * block_dim * ell_idx +
                                    block_dim * (tidy + y) + (tidx + x)]
                         : U{};
        }

        __syncthreads();

        if (block_col >= 0) {
          if ((trans_A == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) &&
              (trans_B == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)) {
            for (T l = 0; l < BELL_BLOCK_DIM; l++) {
              sum += conj(shared_A[BELL_BLOCK_DIM * l + tidx]) *
                     conj(shared_B[BELL_BLOCK_DIM * tidy + l]);
            }
          } else if ((trans_A != ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) &&
                     (trans_B == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)) {
            for (T l = 0; l < BELL_BLOCK_DIM; l++) {
              sum += shared_A[BELL_BLOCK_DIM * l + tidx] *
                     conj(shared_B[BELL_BLOCK_DIM * tidy + l]);
            }
          } else if ((trans_A == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) &&
                     (trans_B != ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)) {
            for (T l = 0; l < BELL_BLOCK_DIM; l++) {
              sum += conj(shared_A[BELL_BLOCK_DIM * l + tidx]) *
                     shared_B[BELL_BLOCK_DIM * tidy + l];
            }
          } else {
            for (T l = 0; l < BELL_BLOCK_DIM; l++) {
              sum += shared_A[BELL_BLOCK_DIM * l + tidx] *
                     shared_B[BELL_BLOCK_DIM * tidy + l];
            }
          }
        }

        __syncthreads();
      }
    }

    const T shift_C = (order_C == ALPHASPARSE_ORDER_COL)
                        ? (global_row + ldc * global_col)
                        : (global_row * ldc + global_col);
    if (block_row < Mb && global_col < N && (tidx + x) < block_dim) {
      if (beta == W{}) {
        C[shift_C] = alpha * sum;
      } else {
        C[shift_C] = beta * C[shift_C] + alpha * sum;
      }
    }
  }
}

template<typename T, typename U, typename V, typename W>
static alphasparseStatus_t
spmm_bell(alphasparseHandle_t handle,
          alphasparseOperation_t trans_A,
          alphasparseOperation_t trans_B,
          alphasparseOrder_t order_B,
          alphasparseOrder_t order_C,
          alphasparseDirection_t dir_A,
          T mb,
          T n,
          W alpha,
          T bell_cols,
          T block_dim,
          const T* __restrict__ bell_col_ind,
          const U* __restrict__ bell_val,
          const U* __restrict__ B,
          T ldb,
          W beta,
          V* __restrict__ C,
          T ldc,
          alphasparseIndexBase_t idx_base)
{
  const T BLOCKSIZE = 32;
  const T WF_SIZE = 32;
  dim3 bellmm_blocks((mb - 1) / 1 + 1, (n - 1) / 32 + 1);
  dim3 bellmm_threads(32, 32, 1);
  bellmm_general_blockdim_device<BLOCKSIZE, WF_SIZE, T, U, V, W>
    <<<bellmm_blocks, bellmm_threads, 0, handle->stream>>>(trans_A,
                                                           trans_B,
                                                           order_B,
                                                           order_C,
                                                           dir_A,
                                                           mb,
                                                           n,
                                                           alpha,
                                                           bell_cols,
                                                           block_dim,
                                                           bell_col_ind,
                                                           bell_val,
                                                           B,
                                                           ldb,
                                                           beta,
                                                           C,
                                                           ldc,
                                                           idx_base);

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
