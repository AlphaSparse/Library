#include "alphasparse/kernel_plain.h"
#include "alphasparse/util.h"
#include "alphasparse/util/malloc.h"
#include <memory.h>

#include "../../../../hygon/format/transpose_csr.hpp"
template <typename J>
alphasparseStatus_t trsv_csr_n_hi_trans_plain(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_csr<J>(A, &matB);
    return trsv_csr_n_lo_plain(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}

template <typename J>
alphasparseStatus_t trsv_csr_u_hi_trans_plain(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_csr<J>(A, &matB);
    return trsv_csr_u_lo_plain(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}

template <typename J>
alphasparseStatus_t trsv_csr_n_lo_trans_plain(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_csr<J>(A, &matB);
    return trsv_csr_n_hi_plain(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}

template <typename J>
alphasparseStatus_t trsv_csr_u_lo_trans_plain(const J alpha, 
                          const internal_spmat A,
                          const J *x, 
                          J *y)
{    
    //创建B并获取A的转置
    internal_spmat matB;
    transpose_csr<J>(A, &matB);
    return trsv_csr_u_hi_plain(alpha, matB -> rows, matB -> cols, matB->nnz, matB->row_data, matB->row_data + 1, matB->col_data,  (J*)(matB->val_data), x, y);
}