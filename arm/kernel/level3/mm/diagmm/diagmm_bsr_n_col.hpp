#include "alphasparse/compute.h"
#include "alphasparse/util.h"
#include <memory.h>

template <typename J>
alphasparseStatus_t diagmm_bsr_n_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT block_rowA = mat->rows;
    ALPHA_INT rowA = mat->rows * mat->block_dim;
    J diag[rowA]; //�洢�Խ�Ԫ��
    memset(diag, '\0', sizeof(J) * rowA);
    ALPHA_INT bs = mat->block_dim;
    
    for (ALPHA_INT ar = 0; ar < block_rowA; ++ar)
    {
        for (ALPHA_INT ai = mat->row_data[ar]; ai < mat->row_data[ar+1]; ++ai)
        {
            if (mat->col_data[ai] == ar) //�Խǿ�
            {
                //diag[ar] = ((J *)mat->val_data)[ai];
                for(ALPHA_INT block_i = 0; block_i < bs; block_i++) //���ʿ��ڶԽ�Ԫ��
                {
                    diag[ar*bs+block_i] = ((J *)mat->val_data)[ai*bs*bs + block_i*bs + block_i];
                }
            } 
        }   
    }
    
    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            //y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * diag[cr] * x[index2(cc, cr, ldx)];
            J t1, t2;
            t1 = alpha_mul(beta, y[index2(cc, cr, ldy)]);
            t2 = alpha_mul(alpha, diag[cr]);
            t2 = alpha_mul(t2, x[index2(cc, cr, ldx)]);
            y[index2(cc, cr, ldy)] = alpha_add(t1, t2);
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
