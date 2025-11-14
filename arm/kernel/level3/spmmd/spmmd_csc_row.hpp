#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"

#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t spmmd_csc_row(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    // if (matA->cols != matB->rows)
    //     return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    ALPHA_INT m = matA->rows;
    ALPHA_INT n = matB->cols;

    //memset(matC, '\0', m * ldc * sizeof(J));
    if(ldc == matB->cols)
    {
        memset(matC, '\0', matA->rows * ldc * sizeof(J));
    }
    else
    {
        for(ALPHA_INT i = 0; i < matA->rows; i++)
        {
            for(ALPHA_INT j = 0; j < matB->cols; j++)
            {
                matC[index2(i, j, ldc)] = alpha_setzero(matC[index2(i, j, ldc)]);
            }
        }
    }

    // ����
    for (ALPHA_INT bc = 0; bc < n; bc++)
    {
        for (ALPHA_INT bi = matB->col_data[bc]; bi < matB->col_data[bc+1]; bi++)
        {
            ALPHA_INT ac = matB->row_data[bi]; // ac == br
            //J bv = ((J *)matB->val_data)[bi]; // bv := B[br][bc]
            J bv;
            bv = ((J *)matB->val_data)[bi];
            for (ALPHA_INT ai = matA->col_data[ac]; ai < matA->col_data[ac+1]; ai++)
            {
                ALPHA_INT ar = matA->row_data[ai];
                //J av = ((J *)matA->val_data)[ai];
                //matC[index2(ar, bc, ldc)] += av * bv;
                J av;
                av = ((J *)matA->val_data)[ai];
                J tmp;
                tmp = alpha_mul(av, bv);
                matC[index2(ar, bc, ldc)] = alpha_add(matC[index2(ar, bc, ldc)], tmp);
                // matC[index2(ar, bc, ldc)].real += tmp.real;
                // matC[index2(ar, bc, ldc)].imag += tmp.imag;
            }
        }
    }
    
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
