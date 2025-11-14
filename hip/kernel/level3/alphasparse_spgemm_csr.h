#include <hip/hip_runtime.h>
#include "alphasparse.h"
#include <iostream>

template<typename T, typename U>
__global__ static void
spgemm_blk(T m,
           T n,
           T k,
           const U alpha,
           const U *csr_val_A,
           const T *csr_row_ptr_A,
           const T *csr_col_ind_A,
           const U *csr_val_B,
           const T *csr_row_ptr_B,
           const T *csr_col_ind_B,
           const U beta,
           U *csr_val_C,
           const T *csr_row_ptr_C,
           T *csr_col_ind_C)
{
    T tid    = hipThreadIdx_x;
    T stride = hipBlockDim_x;

    extern __shared__ char shr[];
    U *values  = reinterpret_cast<U *>(shr);
    T *write_back = reinterpret_cast<T *>(values + n);

    //for (T ar = tid; ar < m; ar += stride)
    T ar = hipBlockIdx_x;
    {
        for (T i = tid; i < n; i += stride) {
            values[i] = U{};
            write_back[i] = T{};
        }
        __syncthreads();

        for (T ai = csr_row_ptr_A[ar] + tid; ai < csr_row_ptr_A[ar + 1]; ai += stride) {
            T br = csr_col_ind_A[ai];
            U tmp = alpha * csr_val_A[ai];

            for (T bi = csr_row_ptr_B[br]; bi < csr_row_ptr_B[br + 1]; bi++) {
                T bc    = csr_col_ind_B[bi];
                U bv = csr_val_B[bi];

                //values[bc] += tmp * bv;
                U t = tmp * bv;
                atomicAdd(&values[bc], t);

                write_back[bc] = 1;
            }
        }
        __syncthreads();

        // in-place prefix sum
        T n64    = 1;
        T stop   = 2;
        T t_stop = 1;
        T i;

        while (n64 < n)
            n64 = n64 << 1;
        n64 = n64 >> 1;

        if (n64 != 0) {
            while (stop <= n64) {
                for (i = tid; i < n64; i += stride) {
                    if (i % stop >= t_stop)
                        write_back[i] += write_back[i - i % t_stop - 1];
                }
                __syncthreads();

                stop   = stop << 1;
                t_stop = t_stop << 1;
            }
        } else
            n64++;

        if (tid == 0) {
            for (T i = n64; i < n; i++) {
                write_back[i] = write_back[i] + write_back[i - 1];
            }
        }

        __syncthreads();

        T index = csr_row_ptr_C[ar];
        for (T c = tid; c < n; c += stride) {
            if (c == 0 && write_back[c]) {
                csr_col_ind_C[index] = c;
                csr_val_C[index]     = values[c];
            } else {
                if (write_back[c] - write_back[c - 1]) {
                    csr_col_ind_C[index + write_back[c] - 1] = c;
                    csr_val_C[index + write_back[c] - 1]     = values[c];
                }
            }
        }

        //TODO scan + write
        // if (tid == 0)
        // {
        //     T index = csr_row_ptr_C[ar];
        //     for (T c = 0; c < n; c++)
        //     {
        //         if (write_back[c])
        //         {
        //             csr_col_ind_C[index] = c;
        //             csr_val_C[index] = values[c];
        //             index += 1;
        //         }
        //     }
        // }
    }
}


#define HASH_SIZE 256
#define EMPTY -1

template<typename T, typename U>
__global__ static void
spgemm_hash(T m,
           T n,
           T k,
           const U alpha,
           const U* csr_val_A,
           const T* csr_row_ptr_A,
           const T* csr_col_ind_A,
           const U* csr_val_B,
           const T* csr_row_ptr_B,
           const T* csr_col_ind_B,
           const U beta,
           U* csr_val_C,
           const T* csr_row_ptr_C,
           T* csr_col_ind_C)
{
    const int row = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if (row >= m) return;

    int hash_col[HASH_SIZE];
    U hash_val[HASH_SIZE];
    for (int i = 0; i < HASH_SIZE; i++) {
        hash_col[i] = EMPTY;
        hash_val[i] = U{};
    }
    const int c_start = csr_row_ptr_C[row];
    const int row_start_A = csr_row_ptr_A[row];
    const int row_end_A = csr_row_ptr_A[row + 1];

    for (int j = row_start_A; j < row_end_A; j++) {
        const int col_A = csr_col_ind_A[j];
        U val_A = alpha * csr_val_A[j];
        const int row_start_B = csr_row_ptr_B[col_A];
        const int row_end_B = csr_row_ptr_B[col_A + 1];

        for (int k = row_start_B; k < row_end_B; k++) {
            const int col_B = csr_col_ind_B[k];
            const U val_B = csr_val_B[k];
            U product = val_A * val_B;
            int hash_index = col_B & (HASH_SIZE - 1);
            int cc = 0;
            while (true) {
                if (hash_col[hash_index] == EMPTY) {
                    hash_col[hash_index] = col_B;
                    hash_val[hash_index] = product;
                    
                    break;
                } else if (hash_col[hash_index] == col_B) {
                    hash_val[hash_index] += product;
                    break;
                } else {
                    hash_index = (hash_index + 1) & (HASH_SIZE - 1);
                }
                cc ++;
                assert(cc < HASH_SIZE);
            }
        }
    }
    
    int count = 0;
    // hash table compression
    for (int i = 0; i < HASH_SIZE; i++) {
        if (hash_col[i] != EMPTY) {            
            if (count != i) {
                hash_col[count] = hash_col[i];
                hash_val[count] = hash_val[i];
            }
            count++;
        }
    }      

    // sort compression array
    for (int i = 1; i < count; i++) {
        int key_col = hash_col[i];
        U key_val = hash_val[i];
        int j = i - 1;
        
        // insert sort
        while (j >= 0 && hash_col[j] > key_col) {
            hash_col[j + 1] = hash_col[j];
            hash_val[j + 1] = hash_val[j];
            j--;
        }
        hash_col[j + 1] = key_col;
        hash_val[j + 1] = key_val;
    }

    for (int i = 0; i < count; i++) {
        csr_col_ind_C[c_start + i] = hash_col[i];
        csr_val_C[c_start + i] = hash_val[i];
    }  
}

template<typename T, typename U>
alphasparseStatus_t
spgemm_csr(alphasparseHandle_t handle,
           T m,
           T n,
           T k,
           const U alpha, 
           const T* csr_row_ptr_A,
           const T* csr_col_ind_A,
           const U* csr_val_A,
           T nnz_A,   
           const T* csr_row_ptr_B,
           const T* csr_col_ind_B,
           const U* csr_val_B,
           T nnz_B,
           U beta,           
           T* csr_row_ptr_C,
           T* csr_col_ind_C,
           U* csr_val_C,
           void * externalBuffer2)
{

  const T threadPerBlock = 256;
  const T blockPerGrid = (m - 1) / threadPerBlock + 1;

  spgemm_blk<<<m,
               threadPerBlock,
               n * (sizeof(U) + sizeof(T)), //for blk
               handle->stream>>>(m,
                                 n,
                                 k,
                                 alpha,
                                 csr_val_A,
                                 csr_row_ptr_A,
                                 csr_col_ind_A,
                                 csr_val_B,
                                 csr_row_ptr_B,
                                 csr_col_ind_B,
                                 beta,
                                 csr_val_C,
                                 csr_row_ptr_C,
                                 csr_col_ind_C); 

  return ALPHA_SPARSE_STATUS_SUCCESS;
}
