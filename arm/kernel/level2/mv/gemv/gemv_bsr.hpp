#include "alphasparse/kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/util.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <string.h>

#include "stdio.h"
template <typename TYPE>
alphasparseStatus_t gemv_bsr_for_each_thread(const TYPE alpha, const internal_spmat A,
                                                    const TYPE *x, const TYPE beta,
                                                    TYPE *y, ALPHA_INT lrs, ALPHA_INT lre) {
  ALPHA_INT bs = A->block_dim;
  ALPHA_INT m_inner = A->rows;
  ALPHA_INT n_inner = A->cols;
  ALPHA_INT task_rows = (lre - lrs) * bs;
  // For matC, block_layout is defaulted as row_major
  TYPE *tmp = (TYPE *)alpha_malloc(sizeof(TYPE) * task_rows);
  memset(tmp, 0, sizeof(TYPE) * task_rows);
  if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR) {
    for (ALPHA_INT i = lrs, j = 0; i < lre; i++, j++) {
      for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++) {
        // TODO Code here if unroll is needed
        TYPE *res = &tmp[bs * j];
        TYPE *val = &((TYPE *)A->val_data)[ai * bs * bs];
        const TYPE *rhs = &x[bs * A->col_data[ai]];
        for (int r = 0; r < bs; r++) {
          for (int k = 0; k < bs; k++) {
            res[r] = alpha_madd(val[r * bs + k], rhs[k], res[r]);
          }
        }
      }
    }
  }
  // For Fortran, block_layout is defaulted as col_major
  else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR) {
    for (ALPHA_INT i = lrs, j = 0; i < lre; i++, j++) {
      for (ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++) {
        TYPE *res = &tmp[bs * j];
        TYPE *val = &((TYPE *)A->val_data)[ai * bs * bs];
        const TYPE *rhs = &x[bs * A->col_data[ai]];
        for (int r = 0; r < bs; r++) {
          for (int k = 0; k < bs; k++) {
            res[r] = alpha_madd(val[r + k * bs], rhs[k], res[r]);
          }
        }
      }
    }
  } else
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;

  for (ALPHA_INT m = lrs * bs, m_t = 0; m < lre * bs; m++, m_t++) {
    y[m] = alpha_mul(y[m], beta);
    y[m] = alpha_madd(tmp[m_t], alpha, y[m]);
  }
  free(tmp);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_bsr_omp(const TYPE alpha, const internal_spmat A,
                                 const TYPE *x, const TYPE beta, TYPE *y) {
  ALPHA_INT m_inner = A->rows;
  ALPHA_INT thread_num = alpha_get_thread_num();

  ALPHA_INT partition[thread_num + 1];
  // TYPE *temp = (TYPE*)malloc(thread_num*sizeof(TYPE));
  // memset(temp, 0, thread_num*sizeof(TYPE));
  balanced_partition_row_by_nnz(A->row_data + 1, m_inner, thread_num, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    ALPHA_INT tid = alpha_get_thread_id();
    ALPHA_INT local_m_s = partition[tid];
    ALPHA_INT local_m_e = partition[tid + 1];
    gemv_bsr_for_each_thread(alpha, A, x, beta, y, local_m_s, local_m_e);
  }
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_bsr(const TYPE alpha, const internal_spmat A, const TYPE *x,
                          const TYPE beta, TYPE *y) {
  return gemv_bsr_omp(alpha, A, x, beta, y);
}

template <typename TYPE>
alphasparseStatus_t gemv_bsr_trans(const TYPE alpha,
                       const internal_spmat A,
                       const TYPE* x,
                       const TYPE beta,
                       TYPE* y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	ALPHA_INT n_inner = A->cols;

	 TYPE temp;
	temp = alpha_setzero(temp);
	// y = y * beta
	for(ALPHA_INT m = 0; m < A->cols * A->block_dim; m++){
		y[m] = alpha_mul(y[m], beta);
		//y[m] *= beta;
	}
	// For matC, block_layout is defaulted as row_major
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for(ALPHA_INT i = 0; i < m_inner; i++){
			for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1];ai++){
				// block[ai]: [A->col_data[ai]][i]
				for(ALPHA_INT col_inner = 0; col_inner < bs; col_inner++){
					for(ALPHA_INT row_inner = 0; row_inner < bs; row_inner++){
						temp = alpha_mul(alpha, ((TYPE *)A->val_data)[ai*bs*bs+row_inner+col_inner*bs]);
						temp = alpha_mul(temp, x[bs*i+col_inner]);
						y[bs*A->col_data[ai]+row_inner] = alpha_add(y[bs*A->col_data[ai]+row_inner], temp);
                        //y[bs*A->col_data[ai]+row_inner] += alpha*((TYPE *)A->val_data)[ai*bs*bs+row_inner+col_inner*bs]*x[bs*i+col_inner];
					}
				// over for block	
				}
			}
		}
	}
	// For Fortran, block_layout is defaulted as col_major
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for(ALPHA_INT i = 0; i < m_inner; i++){
			for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1];ai++){
				// block[ai]: [A->col_data[ai]][i]
				for(ALPHA_INT row_inner = 0; row_inner < bs; row_inner++){
					for(ALPHA_INT col_inner = 0; col_inner < bs; col_inner++){
						temp = alpha_mul(alpha, ((TYPE *)A->val_data)[ai*bs*bs+col_inner+row_inner*bs]);
						temp = alpha_mul(temp, x[bs*i+col_inner]);
						y[bs*A->col_data[ai]+row_inner] = alpha_add(y[bs*A->col_data[ai]+row_inner], temp);
						//y[bs*A->col_data[ai]+row_inner] += alpha*((TYPE *)A->val_data)[ai*bs*bs+col_inner+row_inner*bs]*x[bs*i+col_inner];
					}
				// over for block	
				}
			}
		}
	}
    else return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename TYPE>
alphasparseStatus_t gemv_bsr_conj(const TYPE alpha,
                       const internal_spmat A,
                       const TYPE* x,
                       const TYPE beta,
                       TYPE* y)
{
	ALPHA_INT bs = A->block_dim;
	ALPHA_INT m_inner = A->rows;
	//m_inner = ( m_inner*bs == A->rows)?(m_inner):(m_inner+1);
	ALPHA_INT n_inner = A->cols;
	//n_inner = ( n_inner*bs == A->cols)?(n_inner):(n_inner+1);

	// y = y * beta
	for(ALPHA_INT m = 0; m < A->cols * A->block_dim; m++){
		y[m] = alpha_mul(y[m], beta);
		//y[m] *= beta;
	}
	TYPE temp;
	temp = alpha_setzero(temp);
	// For matC, block_layout is defaulted as row_major
	if (A->block_layout == ALPHA_SPARSE_LAYOUT_ROW_MAJOR){
		for(ALPHA_INT i = 0; i < m_inner; i++){
			for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1]; ai++){
				// block[ai]: [i][A->col_data[ai]]
				for(ALPHA_INT row_inner = 0; row_inner < bs; row_inner++){
					for(ALPHA_INT col_inner = 0; col_inner < bs; col_inner++){
						TYPE cv = ((TYPE *)A->val_data)[ai*bs*bs+row_inner*bs+col_inner];
						cv = cmp_conj(cv);
						temp = alpha_mul(alpha, cv);
						temp = alpha_mul(temp, x[ bs * i+row_inner ]);
						y[bs * A->col_data[ai]+col_inner ] = alpha_add(y[bs * A->col_data[ai] + col_inner ], temp);
						//y[bs*i+row_inner] += alpha*((TYPE *)A->val_data)[ai*bs*bs+row_inner*bs+col_inner]*x[bs*A->col_data[ai]+col_inner];
					}
				// over for block	
				}
			}
		}
	}
	// For Fortran, block_layout is defaulted as col_major
	else if (A->block_layout == ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR){
		for(ALPHA_INT i = 0; i < m_inner; i++){
			for(ALPHA_INT ai = A->row_data[i]; ai < A->row_data[i+1];ai++){
				// block[ai]: [i][A->cols[ai]]
				for(ALPHA_INT col_inner = 0; col_inner < bs; col_inner++){
					for(ALPHA_INT row_inner = 0; row_inner < bs; row_inner++){
						TYPE cv = ((TYPE *)A->val_data)[ai*bs*bs+col_inner*bs+row_inner];
						cv = cmp_conj(cv);
						temp = alpha_mul(alpha, cv);
						temp = alpha_mul(temp, x[bs*i+row_inner ]);
						y[bs * A->col_data[ai] + col_inner] = alpha_add(y[bs * A->col_data[ai] + col_inner], temp);
						//y[bs*i+row_inner] += alpha*((TYPE *)A->val_data)[ai*bs*bs+col_inner*bs+row_inner]*x[bs*A->col_data[ai]+col_inner];
					}
				// over for block	
				}
			}
		}
	}
	else return ALPHA_SPARSE_STATUS_INVALID_VALUE;

    return ALPHA_SPARSE_STATUS_SUCCESS;
}