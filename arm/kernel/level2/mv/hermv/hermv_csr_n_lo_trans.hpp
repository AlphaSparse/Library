#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"

template <typename TYPE>
alphasparseStatus_t
hermv_csr_n_lo_trans(const TYPE alpha,
      const internal_spmat A,
      const TYPE *x,
      const TYPE beta,
      TYPE *y)
{
    /*internal_spmat transposed_mat;
    transpose_csr(A, &transposed_mat);
    template <typename TYPE>
alphasparseStatus_t status = hermv_c_csr_n_hi(alpha, transposed_mat, x, beta, y);
    destroy_csr(transposed_mat); //inf-norm = 0.0935797095298767
    return status;*/
    const ALPHA_INT m = A->rows;
    const ALPHA_INT n = A->cols;
    

    for(ALPHA_INT i = 0; i < m; ++i)
    {
        // y[i] *= beta;
        y[i] = alpha_mul(y[i], beta); 
        for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ++ai)
        {
            const ALPHA_INT col = A->col_data[ai];
            TYPE tmp;
			tmp = alpha_setzero(tmp);
            if(col < i)
            {
                TYPE conval;
				conval = cmp_conj(conval);
                tmp = alpha_mul(((TYPE*)A->val_data)[ai], x[i]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[col] = alpha_add(y[col], tmp);
				
                tmp = alpha_mul(conval, x[col]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[i] = alpha_add(y[i], tmp);
                // y[col] += alpha * ((TYPE*)A->val_data)[ai] * x[i];
                // y[i] += alpha * ((TYPE*)A->val_data)[ai] * x[col];                                 
            }
			else if(col == i)
			{
                TYPE conval;
				conval = cmp_conj(conval);
                tmp = alpha_mul(conval, x[col]); 
            	tmp = alpha_mul(alpha, tmp); 
            	y[i] = alpha_add(y[i], tmp);
                // y[i] += alpha * ((TYPE*)A->val_data)[ai] * x[col];                                 
			}
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS; //inf-norm = 0.0935796424746513
}
