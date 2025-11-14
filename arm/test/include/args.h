#pragma once

#include <getopt.h>
#include <stdlib.h>
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"

#include <stdbool.h>

// #ifndef __MKL__
// #define __MKL__
// #endif  

#define DEFAULT_DATA_FILE ""
#define DEFAULT_THREAD_NUM 1
#define DEFAULT_CHECK false
#define DEFAULT_WARM false
#define DEFAULT_LAYOUT ALPHA_SPARSE_LAYOUT_ROW_MAJOR
#define DEFAULT_SPARSE_OPERATION ALPHA_SPARSE_OPERATION_NON_TRANSPOSE
#define DEFAULT_ITER 1
#define DEFAULT_PARAM 4
#define DEFAULT_FORMAT ALPHA_SPARSE_FORMAT_CSR
#define DEFAULT_DATA_TYPE ALPHA_SPARSE_DATATYPE_FLOAT

alphasparse_layout_t alphasparse_layout_parse(const char *arg);
alphasparseFormat_t alphasparse_format_parse(const char *arg);
alphasparse_datatype_t alphasparse_data_type_parse(const char *arg);
alphasparseOperation_t alphasparse_operation_parse(const char *arg);
alphasparse_matrix_type_t alphasparse_matrix_type_parse(const char *arg);
alphasparse_fill_mode_t alphasparse_fill_mode_parse(const char *arg);
alphasparse_diag_type_t alphasparse_diag_type_parse(const char *arg);

void args_help(const int argc, const char *argv[]);
bool args_get_if_check(const int argc, const char *argv[]);
bool args_get_if_warm(const int argc, const char *argv[]);
int args_get_thread_num(const int argc, const char *argv[]);
int args_get_columns(const int argc, const char *argv[], int k);
int args_get_iter(const int argc, const char *argv[]);
int args_get_param0(const int argc, const char *argv[]);
int args_get_param1(const int argc, const char *argv[]);

const char* args_get_data_file(const int argc, const char *argv[]);
const char* args_get_data_fileA(const int argc, const char *argv[]);
const char* args_get_data_fileB(const int argc, const char *argv[]);
const char *args_get_data_fileC(const int argc, const char *argv[]);

alphasparse_layout_t alpha_args_get_layout_helper(const int argc, const char *argv[], const int layout_opt);
struct alpha_matrix_descr alpha_args_get_matrix_descr_helper(const int argc, const char *argv[], const int type_opt, const int fill_opt, const int diag_opt);
alphasparseOperation_t alpha_args_get_trans_helper(const int argc, const char *argv[], const int trans_opt);
alphasparseFormat_t alpha_args_get_format_helper(const int argc, const char *argv[], int);
alphasparse_datatype_t alpha_args_get_data_type_helper(const int argc, const char *argv[], int);

alphasparse_layout_t alpha_args_get_layout(const int argc, const char *argv[]);
alphasparse_layout_t alpha_args_get_layoutB(const int argc, const char *argv[]);
alphasparse_layout_t alpha_args_get_layoutC(const int argc, const char *argv[]);

struct alpha_matrix_descr alpha_args_get_matrix_descrA(const int argc, const char *argv[]);
struct alpha_matrix_descr alpha_args_get_matrix_descrB(const int argc, const char *argv[]);

alphasparseOperation_t alpha_args_get_transA(const int argc, const char *argv[]);
alphasparseOperation_t alpha_args_get_transB(const int argc, const char *argv[]);

alphasparseFormat_t alpha_args_get_format(const int argc, const char *argv[]);
alphasparseFormat_t alpha_args_get_formatA(const int argc, const char *argv[]);
alphasparseFormat_t alpha_args_get_formatB(const int argc, const char *argv[]);

alphasparse_datatype_t alpha_args_get_data_type(const int argc, const char *argv[]);

void alpha_arg_parse(const int argc, const char *argv[], alphasparse_layout_t *layout, alphasparseOperation_t *transA, alphasparseOperation_t *transB, struct alpha_matrix_descr *descr);

#ifdef __MKL__
#include <mkl.h>

sparse_layout_t mkl_sparse_layout_parse(const char *arg);
sparse_operation_t mkl_sparse_operation_parse(const char *arg);
sparse_matrix_type_t mkl_sparse_matrix_type_parse(const char *arg);
sparse_fill_mode_t mkl_sparse_fill_mode_parse(const char *arg);
sparse_diag_type_t mkl_sparse_diag_type_parse(const char *arg);

sparse_layout_t mkl_args_get_layout_helper(const int argc, const char *argv[], const int layout_opt);
struct matrix_descr mkl_args_get_matrix_descr_helper(const int argc, const char *argv[], const int type_opt, const int fill_opt, const int diag_opt);
sparse_operation_t mkl_args_get_trans_helper(const int argc, const char *argv[], const int trans_opt);

sparse_layout_t mkl_args_get_layout(const int argc, const char *argv[]);
sparse_layout_t mkl_args_get_layoutB(const int argc, const char *argv[]);
sparse_layout_t mkl_args_get_layoutC(const int argc, const char *argv[]);

struct matrix_descr mkl_args_get_matrix_descrA(const int argc, const char *argv[]);
struct matrix_descr mkl_args_get_matrix_descrB(const int argc, const char *argv[]);

sparse_operation_t mkl_args_get_transA(const int argc, const char *argv[]);
sparse_operation_t mkl_args_get_transB(const int argc, const char *argv[]);
#endif
