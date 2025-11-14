#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/util/malloc.h"
#include <memory.h>
#include "alphasparse/compute.h"

#include "../../../../hygon/format/transpose_conj_csr.hpp"
template <typename J>
alphasparseStatus_t trsv_csr_n_hi_conj(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_conj_csr<J>(A, &matB);
    return trsv_csr_n_lo(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}

template <typename J>
alphasparseStatus_t trsv_csr_u_hi_conj(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_conj_csr<J>(A, &matB);
    return trsv_csr_u_lo(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}

template <typename J>
alphasparseStatus_t trsv_csr_n_lo_conj(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_conj_csr<J>(A, &matB);
    return trsv_csr_n_hi(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}

template <typename J>
alphasparseStatus_t trsv_csr_u_lo_conj(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_conj_csr<J>(A, &matB);
    return trsv_csr_u_hi(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}