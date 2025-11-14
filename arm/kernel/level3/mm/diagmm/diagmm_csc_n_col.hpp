#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_csc_n_col(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    // ϡ�������Գ��ܾ���
    // y := alpha*A*x + beta*y
    ALPHA_INT rowA = mat->rows;
    ALPHA_INT colA = mat->cols;
    J diag[colA]; //�洢�Խ�Ԫ��
    for (ALPHA_INT ac = 0; ac < colA; ++ac) //��mat�ĶԽ�Ԫ����ȡ����������diag
    {
        diag[ac] = alpha_setzero(diag[ac]);
        for (ALPHA_INT ai = mat->col_data[ac]; ai < mat->col_data[ac+1]; ++ai)
            if (mat->row_data[ai] == ac)
            {
                //diag[ac] = ((J *)mat->val_data)[ai];
                diag[ac] = ((J *)mat->val_data)[ai];
            }
    }

    for (ALPHA_INT cc = 0; cc < columns; ++cc)
        for (ALPHA_INT cr = 0; cr < rowA; ++cr)
        {
            //y[index2(cc, cr, ldy)] = beta * y[index2(cc, cr, ldy)] + alpha * diag[cr] * x[index2(cc, cr, ldx)];
            J temp1, temp2;
            temp1 = alpha_mul(beta, y[index2(cc, cr, ldy)]);
            temp2 = alpha_mul(diag[cr], x[index2(cc, cr, ldx)]);
            temp2 = alpha_mul(alpha, temp2);
            y[index2(cc, cr, ldy)] = alpha_add(temp1, temp2);
        }
            
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
