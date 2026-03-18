#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"


template <typename TYPE>
alphasparseStatus_t gemv_csc_plain(const TYPE alpha, const internal_spmat A, const TYPE *x,
                          const TYPE beta, TYPE *y) {
      const ALPHA_INT m = A->rows;
      const ALPHA_INT n = A->cols;
      for (ALPHA_INT j = 0; j < m; ++j)
      {
        y[j] = alpha_mul(y[j], beta); 
      }
      TYPE tmp;
      tmp = alpha_setzero(tmp);     
      for (ALPHA_INT i = 0; i < n; i++)
      {
          for (ALPHA_INT ai = A->col_data[i]; ai < A->col_data[i+1]; ai++)
          {
              tmp = alpha_mul(((TYPE*)A->val_data)[ai], x[i]); 
              tmp = alpha_mul(alpha, tmp); 
              y[A->row_data[ai]] = alpha_add(y[A->row_data[ai]], tmp);
          } 
    }
      
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_csc_conj_plain(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE *x,
		               const TYPE beta,
		               TYPE *y)
{
    ALPHA_INT m = A->cols;
    for (ALPHA_INT r = 0; r < m; r++)
    {
        y[r] = alpha_mul(y[r], beta); 
        TYPE tmp;
        tmp = alpha_setzero(tmp);        
        
        for (ALPHA_INT ai = A->col_data[r]; ai < A->col_data[r+1]; ai++)
        {            
            TYPE inner_tmp;
            // alpha_setzero(inner_tmp);
            inner_tmp = cmp_conj(((TYPE *)A->val_data)[ai]);
            inner_tmp = alpha_mul(inner_tmp, x[A->row_data[ai]]); 
            tmp = alpha_add(tmp, inner_tmp);
        }
        tmp = alpha_mul(alpha, tmp); 
        y[r] = alpha_add(y[r], tmp); 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_csc_trans_plain(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE *x,
		               const TYPE beta,
		               TYPE *y)
{
    ALPHA_INT m = A->cols;
    for (ALPHA_INT r = 0; r < m; r++)
    {
        y[r] = alpha_mul(y[r], beta); 
        TYPE tmp;
        tmp = alpha_setzero(tmp);        
        
        for (ALPHA_INT ai = A->col_data[r]; ai < A->col_data[r+1]; ai++)
        {            
            TYPE inner_tmp;
            inner_tmp = alpha_setzero(inner_tmp);
            inner_tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[A->row_data[ai]]); 
            tmp = alpha_add(tmp, inner_tmp);
        }
        tmp = alpha_mul(alpha, tmp); 
        y[r] = alpha_add(y[r], tmp); 
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}