#include "alphasparse.h"
#include <hipcub/hipcub.hpp>  

template<typename T, typename U>
alphasparseStatus_t
spgemm_copy_csr(alphasparseHandle_t handle,    
               const T* csr_row_ptr_C,
               const T* csr_col_ind_C,
               U* csr_val_C,
               T nnz_C,
               U beta,
               alphasparseIndexBase_t baseC)
{
    return ALPHA_SPARSE_STATUS_SUCCESS;
}