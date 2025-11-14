#pragma once

/**
 * @brief header for file read and write utils
 */

#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include <hip/hip_complex.h>

#include <stdlib.h>
#include <stdio.h>
#define USE_MTX 0
#define USE_BIN 1

#define BUFFER_SIZE 1024

FILE *alpha_open(const char *filename, const char *modes);
void alpha_close(FILE *stream);

void result_write(const char *path, const size_t ele_num, size_t ele_size, const void *data);

void alpha_read_coo_s(const char *file, int *m_p, int *n_p, int *nnz_p, int **row_index, int **col_index, float **values);
void alpha_read_coo_d(const char *file, int *m_p, int *n_p, int *nnz_p, int **row_index, int **col_index, double **values);
void alpha_read_coo_c(const char *file, int *m_p, int *n_p, int *nnz_p, int **row_index, int **col_index, hipFloatComplex **values);
void alpha_read_coo_z(const char *file, int *m_p, int *n_p, int *nnz_p, int **row_index, int **col_index, hipDoubleComplex **values);
template <typename T>
void alpha_read_coo(const char *file, int *m_p, int *n_p, int *nnz_p, int **row_index, int **col_index, T **values);

void alpha_read_coo_pad_s(const char *file, int *m_p, int *n_p, int pad, int *nnz_p, int **row_index, int **col_index, float **values);
void alpha_read_coo_pad_d(const char *file, int *m_p, int *n_p, int pad, int *nnz_p, int **row_index, int **col_index, double **values);
void alpha_read_coo_pad_c(const char *file, int *m_p, int *n_p, int pad, int *nnz_p, int **row_index, int **col_index, hipFloatComplex **values);
void alpha_read_coo_pad_z(const char *file, int *m_p, int *n_p, int pad, int *nnz_p, int **row_index, int **col_index, hipDoubleComplex **values);

// template <typename T>
// alphasparseStatus_t alphasparseCreateCoo(
//     alphasparse_matrix_t *A,
//     const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
//     const int rows, const int cols, const int nnz, int *row_indx, int *col_indx,
//     T *values);

typedef struct {
  int64_t rows; //总行数
  int64_t cols; //总列数
  int64_t nnzs; //总nnz个数
  int64_t real_nnz; //complex:2, integer/float:1, pattern 0, real:1
  int64_t field_per_nnz; //complex:2, integer/float:1, pattern 0, real:1
  int64_t num_type; //float:0, integer 1;
  int64_t mtx_sym; //general:0, sym:1, Hermitian:2 
  int64_t reserved;
} bin_header;


typedef union {
   double f[2];
   char c[16];
}float64x2_u;
typedef union{
   double f;
   char c[8];
}float64x1_u ;

typedef union{
   int32_t coo[2];
   char c[8];
}int32x2_u;

// void alpha_dump_nnz_feature(const char *file, int *rows, int *cols, int *nnz, double *sparsity, double *avr_nnz_row, int *min_nnz_row, int *max_nnz_row, double *var_nnz_row, int *diags, double *diag_ratio, double *dia_padding_ratio, double *ell_padding_ratio);
#ifdef __MKL__

#include <mkl.h>
void mkl_read_coo(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, float **values);
void mkl_read_coo_d(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, double **values);
void mkl_read_coo_c(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, MKL_Complex8 **values);
void mkl_read_coo_z(const char *file, MKL_INT *m_p, MKL_INT *n_p, MKL_INT *nnz_p, MKL_INT **row_index, MKL_INT **col_index, MKL_Complex16 **values);
#endif