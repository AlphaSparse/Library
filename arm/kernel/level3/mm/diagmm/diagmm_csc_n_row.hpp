#include "alphasparse/compute.h"
#include "alphasparse/util.h"

template <typename J>
alphasparseStatus_t diagmm_csc_n_row(const J alpha, const internal_spmat mat, const J *x, const ALPHA_INT columns, const ALPHA_INT ldx, const J beta, J *y, const ALPHA_INT ldy)
{
    ALPHA_INT colA = mat->cols;
    ALPHA_INT rowC = mat->rows;
    ALPHA_INT colC = columns;
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

    for (ALPHA_INT cr = 0; cr < rowC; ++cr)
        for (ALPHA_INT cc = 0; cc < colC; ++cc)
        {
            //y[index2(cr, cc, ldy)] = beta * y[index2(cr, cc, ldy)] + alpha * diag[cr] * x[index2(cr, cc, ldx)];
            J temp1, temp2;
            temp1 = alpha_mul(beta, y[index2(cr, cc, ldy)]);
            temp2 = alpha_mul(diag[cr], x[index2(cr, cc, ldx)]);
            temp2 = alpha_mul(alpha, temp2);
            y[index2(cr, cc, ldy)] = alpha_add(temp1, temp2);
        }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}
