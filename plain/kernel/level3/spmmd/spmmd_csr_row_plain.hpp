#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/partition.h"

template <typename J>
alphasparseStatus_t spmmd_csr_row_plain(const internal_spmat matA, const internal_spmat matB, J *matC,
                                        const ALPHA_INT ldc) 
{

    if (matA->cols != matB->rows || ldc < matB->cols)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    ALPHA_INT m = matA->rows;

    for(ALPHA_INT i = 0; i < matA->rows; i++)
        for(ALPHA_INT j = 0; j < matB->cols; j++)
        {
            matC[index2(i, j, ldc)] = alpha_setzero(matC[index2(i, j, ldc)]);
        }
        
    for (ALPHA_INT ar = 0; ar < m; ar++)
    {
        for (ALPHA_INT ai = matA->row_data[ar]; ai < matA->row_data[ar+1]; ai++)
        {
            ALPHA_INT br = matA->col_data[ai];
            J av = ((J*)matA->val_data)[ai];
            for (ALPHA_INT bi = matB->row_data[br]; bi < matB->row_data[br+1]; bi++)
            {
                ALPHA_INT bc = matB->col_data[bi];
                J bv = ((J*)matB->val_data)[bi];
                matC[index2(ar, bc, ldc)] = alpha_madde(matC[index2(ar, bc, ldc)], av, bv);
            }
        }
    }
}
