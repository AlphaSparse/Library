
// #define _GNU_SOURCE
#include <memory.h>
#include <sched.h>
#include <iostream> 
#include "alphasparse/util.h"
#include "csrmv/csrmv_kernel.h"
#include "alphasparse/opt.h"
#include "alphasparse/compute.h"
#include "alphasparse/util/bisearch.h"
#include "alphasparse/util/partition.h"
#include <type_traits>

template <typename I, typename W>
alphasparseStatus_t gemv_csr(const float alpha, const I m, const I n,
                           const W* rows_ptr, const W* rows_end, 
                           const W* col_indx, const float *values,
                           const float *x, const float beta,
                           float *y) {

  W num_threads = alpha_get_thread_num();
  W partition[num_threads + 1];
  balanced_partition_row_by_nnz(rows_end, m, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    const I tid = alpha_get_thread_id();
    W local_m_s = partition[tid];
    W local_m_e = partition[tid + 1];
    W Rows = local_m_e - local_m_s;
    // std::cout<<local_m_s<<" "<<local_m_e<<std::endl;
    const W *rows_start = &rows_ptr[local_m_s];
    float *y_local = &y[local_m_s];

    __spmv_csr_serial_host_sse_float(static_cast<const float>(alpha), (const float)beta, Rows, rows_start, col_indx, (float*)values, (float*)x, (float*)y_local);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename W>
alphasparseStatus_t gemv_csr(const double alpha, const I m, const I n,
                           const W* rows_ptr, const W* rows_end, 
                           const W* col_indx, const double *values,
                           const double *x, const double beta,
                           double *y) {

  W num_threads = alpha_get_thread_num();
  W partition[num_threads + 1];
  balanced_partition_row_by_nnz(rows_end, m, num_threads, partition);
#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    const I tid = alpha_get_thread_id();
    W local_m_s = partition[tid];
    W local_m_e = partition[tid + 1];
    W Rows = local_m_e - local_m_s;

    const W *rows_start = &rows_ptr[local_m_s];
    double *y_local = &y[local_m_s];
    
    __spmv_csr_serial_host_avx2_double(alpha, beta, Rows, rows_start,
                                      col_indx, values, x, y_local);
 
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename W>
alphasparseStatus_t gemv_csr(const ALPHA_Complex8 alpha, const I m, const I n,
                           const W* rows_ptr, const W* rows_end, 
                           const W* col_indx, const ALPHA_Complex8 *values,
                           const ALPHA_Complex8 *x, const ALPHA_Complex8 beta,
                           ALPHA_Complex8 *y) {

  W num_threads = alpha_get_thread_num();
  W partition[num_threads + 1];
  balanced_partition_row_by_nnz(rows_end, m, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    const I tid = alpha_get_thread_id();

    W local_m_s = partition[tid];
    W local_m_e = partition[tid + 1];
    W Rows = local_m_e - local_m_s;

    const W *rows_start = &rows_ptr[local_m_s];
    ALPHA_Complex8 *y_local = &y[local_m_s];
  
    __spmv_csr_serial_host_sse_complex_float(
        alpha, beta, Rows, rows_start, col_indx, values, x, y_local);
   
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename W>
alphasparseStatus_t gemv_csr(const ALPHA_Complex16 alpha, const I m, const I n,
                           const W* rows_ptr, const W* rows_end, 
                           const W* col_indx, const ALPHA_Complex16 *values,
                           const ALPHA_Complex16 *x, const ALPHA_Complex16 beta,
                           ALPHA_Complex16 *y) {

  W num_threads = alpha_get_thread_num();
  W partition[num_threads + 1];
  balanced_partition_row_by_nnz(rows_end, m, num_threads, partition);

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
    const I tid = alpha_get_thread_id();

    W local_m_s = partition[tid];
    W local_m_e = partition[tid + 1];
    W Rows = local_m_e - local_m_s;

    const W *rows_start = &rows_ptr[local_m_s];
    ALPHA_Complex16 *y_local = &y[local_m_s];
 
    __spmv_csr_serial_host_sse_complex_double(
        alpha, beta, Rows, rows_start, col_indx, values, x, y_local);
  }

  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
static alphasparseStatus_t gemv_csr_trans_omp(const J alpha, const I m, const I n,
            const W* rows_start, const W* rows_end, 
            const W* col_indx, const J *values,
            const J *x, const J beta,
            J *y) {

  const W thread_num = alpha_get_thread_num();
  W partition[thread_num + 1];
  balanced_partition_row_by_nnz(rows_end, m, thread_num, partition);
  J **tmp = (J **)malloc(sizeof(J *) * thread_num);
#ifdef _OPENMP
#pragma omp parallel num_threads(thread_num)
#endif
  {
    const I tid = alpha_get_thread_id();
    const I local_m_s = partition[tid];
    const I local_m_e = partition[tid + 1];
    tmp[tid] = (J *)malloc(sizeof(J) * n);
    memset(tmp[tid], '\0', sizeof(J) * n);
    for (I i = local_m_s; i < local_m_e; ++i) {
      const J x_r = x[i];
      int pkl = rows_start[i];
      int pke = rows_end[i];
      for (; pkl < pke - 3; pkl += 4)
      {
          tmp[tid][col_indx[pkl]] = alpha_madd( values[pkl], x_r, tmp[tid][col_indx[pkl]]);
          tmp[tid][col_indx[pkl + 1]] = alpha_madd(values[pkl + 1], x_r, tmp[tid][col_indx[pkl + 1]]);
          tmp[tid][col_indx[pkl + 2]] = alpha_madd(values[pkl + 2], x_r, tmp[tid][col_indx[pkl + 2]]);
          tmp[tid][col_indx[pkl + 3]] = alpha_madd(values[pkl + 3], x_r, tmp[tid][col_indx[pkl + 3]]);
      }
      for (; pkl < pke; ++pkl)
      {
          tmp[tid][col_indx[pkl]] = alpha_madd(values[pkl], x_r, tmp[tid][col_indx[pkl]]);
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (I i = 0; i < n; ++i) {
    J tmp_y;
    tmp_y = alpha_setzero(tmp_y);
    for (I j = 0; j < thread_num; ++j) {
      tmp_y = alpha_add(tmp_y, tmp[j][i]);
    }
    y[i] = alpha_mul(y[i], beta);
    y[i] = alpha_madd(alpha, tmp_y, y[i]);
  }
#ifdef _OPENMP
#pragma omp parallel for num_threads(thread_num)
#endif
  for (I i = 0; i < thread_num; ++i) {
    alpha_free(tmp[i]);
  }
  alpha_free(tmp);
  return ALPHA_SPARSE_STATUS_SUCCESS;
}

template <typename I, typename J, typename W>
alphasparseStatus_t gemv_csr_trans(const J alpha, const I m, const I n,
            const W* rows_start, const W* rows_end, 
            const W* col_indx, const J *values,
            const J *x, const J beta,
            J *y) {
  const I thread_num = alpha_get_thread_num();
  return gemv_csr_trans_omp(alpha, m, n, rows_start, rows_end, col_indx, values, x, beta, y);
}

template <typename I, typename J, typename W>
alphasparseStatus_t
gemv_csr_conj(const J alpha, const I m, const I n,
            const W* rows_ptr, const W* rows_end, 
            const W* col_indx, const J *values,
            const J *x, const J beta,
            J *y)
{
    for (I j = 0; j < n; ++j)
    {
        y[j] = alpha_mul(y[j], beta);
    }
    for (I i = 0; i < m; i++)
    {
        for (I ai = rows_ptr[i]; ai < rows_end[i]; ai++)
        {
            J val = values[ai];

            val = cmp_conj(val);
            
            val = alpha_mul(alpha, val);
            y[col_indx[ai]] = alpha_madd(val, x[i], y[col_indx[ai]]);
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}