#include "alphasparse.h"
#include "csrspgemm_device_speck.h"

template <typename IndexType, typename DataType, int BLOCKS_PER_SM, int THREADS_PER_BLOCK, int MAX_DYNAMIC_SHARED, int MAX_STATIC_SHARED>
alphasparseStatus_t spgemm_csr_spECK(alphasparseHandle_t handle,
                        alphasparseOperation_t opA,
                        alphasparseOperation_t opB,
                        const DataType alpha,
                        alphasparseSpMatDescr_t matA,
                        alphasparseSpMatDescr_t matB,
                        const DataType beta,
                        alphasparseSpMatDescr_t matC,
                        void * externalBuffer2)
{
    MultiplyspECKImplementation<IndexType, DataType, BLOCKS_PER_SM, THREADS_PER_BLOCK, MAX_DYNAMIC_SHARED, MAX_STATIC_SHARED>(handle, opA, opB, alpha, matA, matB, beta, matC, (char *)externalBuffer2);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
