#pragma once
// #ifdef __cplusplus
// extern "C" {
// #endif
#define FILE_SOURCE 0
#define FILE_SOURCE_A 1
#define FILE_SOURCE_B 2
#include <stdio.h>
#include <math.h>

#include "args.h"
#include "check.h"
#include "alphasparse/util/malloc.h"
#include "io.h"

#ifdef __MKL__
#include <mkl.h>
#include <mkl_spblas.h>
static const int mkl_bytes_type[] = {sizeof(MKL_INT), 8, 8, sizeof(MKL_Complex16)};
#endif
static const int bytes_type[] = {4, 8, 8, 16};
// only support formatA==formatB==format
typedef struct {
  const char *file;
  const char *fileA;
  const char *fileB;
  int thread_num;
  int iter;
  bool check;
  bool warm;
  // right-hand columns
  int columns;
  alphasparseOperation_t transA;
  alphasparseOperation_t transB;
  alphasparse_layout_t layout;
  alphasparse_datatype_t data_type;
  alphasparseFormat_t format;
  struct alpha_matrix_descr alpha_descr;
#ifdef __MKL__
  sparse_operation_t mkl_transA;
  struct matrix_descr mkl_descr;
  sparse_layout_t mkl_layout;
#endif
  int param0; // reserved
  int param1;
  int param2;
  int param3;
  int param4;
} alpha_common_args_t;

typedef struct {
  int32_t m, k, nnz;
  int32_t *row_index, *col_index;
  char *values;
  alphasparse_datatype_t data_type;
} matrix_data_t;

void parse_args_and_initialize(int argc, const char *argv[], alpha_common_args_t *common_arg);
void alpha_read_coo_wrapper(matrix_data_t *matrix_data, alpha_common_args_t *common_arg, int file_used,
                          int padded_size);
void malloc_random_fill(alphasparse_datatype_t dt, void **x, const size_t len, unsigned int seed);
void alpha_create_coo_wapper(matrix_data_t *matrix_data, alphasparse_datatype_t data_type,
                           alphasparse_matrix_t *output);
void alpha_convert_matrix_wapper(alphasparseFormat_t fmt, struct alpha_matrix_descr descr, alphasparse_layout_t layout,
                               alphasparse_matrix_t input, alphasparse_matrix_t *output,
                               int row_block, int col_block);
void alpha_initialize_alpha_beta(char *alpha_char, char *beta_char, alphasparse_datatype_t data_type);
void check_arm(alphasparse_datatype_t dt, const char *icty_char, const int len,
               const char *icty_plain_char);
void destory_matrix_data(matrix_data_t *matrix_data);

#ifdef __MKL__
void mkl_create_coo_wapper(matrix_data_t *matrix_data, alphasparse_datatype_t data_type,
                           sparse_matrix_t *output);
void mkl_convert_matrix_wapper(alphasparseFormat_t fmt, struct matrix_descr descr, sparse_layout_t layout, 
                               sparse_matrix_t input, sparse_matrix_t *output, int row_block,
                               int col_block);
void mkl_initialize_alpha_beta(char *alpha_char, char *beta_char, alphasparse_datatype_t data_type);
#endif

// #ifdef __cplusplus
// }
// #endif