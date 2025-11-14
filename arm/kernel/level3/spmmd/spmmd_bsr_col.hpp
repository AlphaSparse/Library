#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t spmmd_bsr_col(const internal_spmat matA, const internal_spmat matB, J *matC, const ALPHA_INT ldc)
{
    if (matA->cols != matB->rows || ldc < matA->rows)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    if(matA->block_layout != matB->block_layout)
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    if(matA->block_dim != matB->block_dim) 
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    
    ALPHA_INT bs = matA->block_dim;
    ALPHA_INT m = matA->rows * bs;
    ALPHA_INT n = matB->cols * bs;
    
    // init C
    for(ALPHA_INT i = 0; i < m; i++)
        for(ALPHA_INT j = 0; j < n; j++)
        {
            matC[index2(j, i, ldc)] = alpha_setzero(matC[index2(j, i, ldc)]);
        }
    
    
    ALPHA_INT A_block_cols = matA->cols;
    ALPHA_INT A_block_rows = matA->rows;
    ALPHA_INT B_block_cols = matB->cols;
    ALPHA_INT B_block_rows = matB->rows;
    // ����
    for (ALPHA_INT ar = 0; ar < A_block_rows; ar++)
    {
        for (ALPHA_INT ai = matA->row_data[ar]; ai < matA->row_data[ar+1]; ai++)
        {
            ALPHA_INT br = matA->col_data[ai];
            //J av = ((J *)matA->val_data)[ai];// av��((J *)matA->val_data)[block_dim*block_dim*ai, block_dim*block_dim*ai+block_dim*block_dim]
            for (ALPHA_INT bi = matB->row_data[br]; bi < matB->row_data[br+1]; bi++)
            {
                ALPHA_INT bc = matB->col_data[bi];
                //J bv = ((J *)matB->val_data)[bi]; //bv��((J *)matB->val_data)[block_dim*block_dim*bi: block_dim*block_dim*bi+block_dim*block_dim]
                // ��������һ�����ܾ���˷�
                //matC[index2(ar, bc, ldc)] += av * bv;
                if(matA->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR)
                {
                    // col major
                    for(ALPHA_INT block_ar = 0; block_ar < matA->block_dim; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < matA->block_dim; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < matB->block_dim; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                ALPHA_INT bs = matA->block_dim;
                                J av = ((J *)matA->val_data)[bs*bs*ai + bs*block_ac + block_ar];
                                J bv = ((J *)matB->val_data)[bs*bs*bi + bs*block_bc + block_br];
                                //matC[index2(bc*bs+block_bc, ar*bs+block_ar, ldc)] += av*bv;
                                matC[index2(bc*bs+block_bc, ar*bs+block_ar, ldc)] = alpha_madde(matC[index2(bc*bs+block_bc, ar*bs+block_ar, ldc)], av, bv);
                            }
                        }
                    }
                }
                else
                {
                    // row major
                    for(ALPHA_INT block_ar = 0; block_ar < matA->block_dim; block_ar++)
                    {
                        for(ALPHA_INT block_ac = 0; block_ac < matA->block_dim; block_ac++) //block_aj==block_bi
                        {
                            for(ALPHA_INT block_bc = 0; block_bc < matB->block_dim; block_bc++)
                            {
                                ALPHA_INT ac = br;
                                ALPHA_INT block_br = block_ac;
                                ALPHA_INT bs = matA->block_dim;
                                J av = ((J *)matA->val_data)[bs*bs*ai + bs*block_ar + block_ac];
                                J bv = ((J *)matB->val_data)[bs*bs*bi + bs*block_br + block_bc];
                                //matC[index2(bc*bs+block_bc, ar*bs+block_ar, ldc)] += av*bv;
                                matC[index2(bc*bs+block_bc, ar*bs+block_ar, ldc)] = alpha_madde(matC[index2(bc*bs+block_bc, ar*bs+block_ar, ldc)], av, bv);
                            }
                        }
                    }
                }
            }
        }
    }

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
