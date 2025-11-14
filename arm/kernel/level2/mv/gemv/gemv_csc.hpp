#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif

template <typename TYPE>
alphasparseStatus_t gemv_csc_omp(const TYPE alpha, const internal_spmat A,
                                        const TYPE *x, const TYPE beta, TYPE *y) {
  const ALPHA_INT m = A->rows;
  const ALPHA_INT n = A->cols;
  const ALPHA_INT thread_num = alpha_get_thread_num();
  ALPHA_INT partition[thread_num + 1];
  balanced_partition_row_by_nnz(A->col_data+1, n, thread_num, partition);

  TYPE **tmp = (TYPE **)malloc(sizeof(TYPE *) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    const ALPHA_INT tid = alpha_get_thread_id();

    const ALPHA_INT local_m_s = partition[tid];
    const ALPHA_INT local_m_e = partition[tid + 1];
    tmp[tid] = (TYPE *)malloc(sizeof(TYPE) * m);
    for (ALPHA_INT k = 0; k < m; k++) {
      tmp[tid][k] = alpha_setzero(tmp[tid][k]);
    }
    TYPE *local_y = tmp[tid];
    for (ALPHA_INT i = local_m_s; i < local_m_e; ++i) {
      const TYPE x_r = x[i];
      int pks = A->col_data[i];
      int pke = A->col_data[i+1];
      ALPHA_INT k = pks;
      for (; k < pke - 3; k += 4) {
        local_y[A->row_data[k]] = alpha_madd(((TYPE *)A->val_data)[k], x_r, local_y[A->row_data[k]]);
        local_y[A->row_data[k + 1]] = alpha_madd(((TYPE *)A->val_data)[k + 1], x_r, local_y[A->row_data[k + 1]]);
        local_y[A->row_data[k + 2]] = alpha_madd(((TYPE *)A->val_data)[k + 2], x_r, local_y[A->row_data[k + 2]]);
        local_y[A->row_data[k + 3]] = alpha_madd(((TYPE *)A->val_data)[k + 3], x_r, local_y[A->row_data[k + 3]]);
      }
      for (; k < pke; ++k) {
        local_y[A->row_data[k]] = alpha_madd(((TYPE *)A->val_data)[k], x_r, local_y[A->row_data[k]]);
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (ALPHA_INT i = 0; i < m; ++i) {
    TYPE tmp_y;
    tmp_y = alpha_setzero(tmp_y);
    for (ALPHA_INT j = 0; j < thread_num; ++j) {
      tmp_y = alpha_add(tmp_y, tmp[j][i]);
    }
    y[i] = alpha_mul(beta, y[i]);
    y[i] = alpha_madd(alpha, tmp_y, y[i]);
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (ALPHA_INT i = 0; i < thread_num; ++i) {
    free(tmp[i]);
  }
  free(tmp);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_csc(const TYPE alpha, const internal_spmat A, const TYPE *x,
                          const TYPE beta, TYPE *y) {
  return gemv_csc_omp(alpha, A, x, beta, y);
}

template <typename TYPE>
alphasparseStatus_t gemv_csc_conj(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE *x,
		               const TYPE beta,
		               TYPE *y)
{
    ALPHA_INT m = A->cols;
    for (ALPHA_INT r = 0; r < m; r++)
    {
        // y[r] *= beta;
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
            // tmp += ((TYPE *)A->val_data)[ai] * x[A->col_data[ai]];
        }
        tmp = alpha_mul(alpha, tmp); 
        y[r] = alpha_add(y[r], tmp); 
        // y[r] += alpha * tmp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_csc_trans(const TYPE alpha,
		               const internal_spmat A,
		               const TYPE *x,
		               const TYPE beta,
		               TYPE *y)
{
    ALPHA_INT m = A->cols;
    for (ALPHA_INT r = 0; r < m; r++)
    {
        // y[r] *= beta;
        y[r] = alpha_mul(y[r], beta); 
        TYPE tmp;
        tmp = alpha_setzero(tmp);        
        
        for (ALPHA_INT ai = A->col_data[r]; ai < A->col_data[r+1]; ai++)
        {            
            TYPE inner_tmp;
            inner_tmp = alpha_setzero(inner_tmp);
            inner_tmp = alpha_mul(((TYPE *)A->val_data)[ai], x[A->row_data[ai]]); 
            tmp = alpha_add(tmp, inner_tmp);
            // tmp += ((TYPE *)A->val_data)[ai] * x[A->col_data[ai]];
        }
        tmp = alpha_mul(alpha, tmp); 
        y[r] = alpha_add(y[r], tmp); 
        // y[r] += alpha * tmp;
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}