#include "alphasparse/kernel.h"
extern "C" {
void __spmv_sell_csigma_serial_host_avx_float(
    const float      alpha,
    const float      beta,
    const int       *slice_ptr,
    const int       *col_idx,
    const float     *values,
    const int       *block_max_nnz,
    const float     *x,
    float           *y,
    const int        C,
    const int        sigma,
    const int        slice_start,
    const int        slice_end,
    const int       *row_idx_map);

void __spmv_sell_csigma_serial_host_avx_double(
    const double     alpha,
    const double     beta,
    const int       *slice_ptr,
    const int       *col_idx,
    const double    *values,
    const int       *block_max_nnz,
    const double    *x,
    double          *y,
    const int        C,
    const int        sigma,
    const int        slice_start,
    const int        slice_end,
    const int       *row_idx_map);
}
