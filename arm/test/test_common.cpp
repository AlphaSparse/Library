#include "./include/test_common.h"

#include "alphasparse.h"

// #ifndef __MKL__
// #define __MKL__
// #endif  
void parse_args_and_initialize(int argc, const char *argv[],
                               alpha_common_args_t *common_arg) {
  common_arg->transA = alpha_args_get_transA(argc, argv);
  common_arg->file = args_get_data_file(argc, argv);
  common_arg->fileA = args_get_data_fileA(argc, argv);

  common_arg->thread_num = args_get_thread_num(argc, argv);
  common_arg->check = args_get_if_check(argc, argv);
  common_arg->iter = args_get_iter(argc, argv);
  common_arg->alpha_descr = alpha_args_get_matrix_descrA(argc, argv);
#ifdef __MKL__
  common_arg->mkl_descr = mkl_args_get_matrix_descrA(argc, argv);
  common_arg->mkl_transA = mkl_args_get_transA(argc, argv);
  common_arg->mkl_layout = mkl_args_get_layout(argc, argv);
#endif
  common_arg->alpha_descr = alpha_args_get_matrix_descrA(argc, argv);
  common_arg->data_type = alpha_args_get_data_type(argc, argv);
  common_arg->format = alpha_args_get_format(argc, argv);
  common_arg->layout = alpha_args_get_layout(argc, argv);
  common_arg->warm = args_get_if_warm(argc, argv);
  common_arg->columns = args_get_columns(argc, argv, 1);

  if (common_arg->transA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
    common_arg->fileB = args_get_data_fileB(argc, argv);
  else
    common_arg->fileB = args_get_data_fileA(argc, argv);

  common_arg->param0 = args_get_param0(argc,argv);
  common_arg->param1 = args_get_param1(argc,argv);
  alpha_set_thread_num(common_arg->thread_num);
  printf("thread num : %d\n", common_arg->thread_num);
}

void alpha_read_coo_wrapper(matrix_data_t *matrix_data,
                            alpha_common_args_t *common_arg, int file_used,
                            int padded_size) {
  const char *file;
  if (file_used == FILE_SOURCE)
    file = common_arg->file;
  else if (file_used == FILE_SOURCE_A) {
    file = common_arg->fileA;
  } else if (file_used == FILE_SOURCE_B) {
    file = common_arg->fileB;
  } else {
    file = NULL;
  }

  if (file == NULL) {
    printf("file is empty!\n");
    exit(-1);
  }
  if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    float *values;
    alpha_read_coo_pad(file, &matrix_data->m, &matrix_data->k, padded_size,
                       &matrix_data->nnz, &matrix_data->row_index,
                       &matrix_data->col_index, &values);
    matrix_data->values = (char *)values;
    matrix_data->data_type = ALPHA_SPARSE_DATATYPE_FLOAT;
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    double *values;
    alpha_read_coo_pad_d(file, &matrix_data->m, &matrix_data->k, padded_size,
                         &matrix_data->nnz, &matrix_data->row_index,
                         &matrix_data->col_index, &values);
    matrix_data->values = (char *)values;
    matrix_data->data_type = ALPHA_SPARSE_DATATYPE_DOUBLE;
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    ALPHA_Complex8 *values;
    alpha_read_coo_pad_c(file, &matrix_data->m, &matrix_data->k, padded_size,
                         &matrix_data->nnz, &matrix_data->row_index,
                         &matrix_data->col_index, &values);
    matrix_data->values = (char *)values;
    matrix_data->data_type = ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX;
  } else if (common_arg->data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    ALPHA_Complex16 *values;
    alpha_read_coo_pad_z(file, &matrix_data->m, &matrix_data->k, padded_size,
                         &matrix_data->nnz, &matrix_data->row_index,
                         &matrix_data->col_index, &values);
    matrix_data->values = (char *)values;
    matrix_data->data_type = ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX;
  }
}

void malloc_random_fill(alphasparse_datatype_t dt, void **x, const size_t len,
                        unsigned int seed) {
  char *x_char = (char *)alpha_malloc(bytes_type[dt] * len);
  *x = x_char;
  if (dt == ALPHA_SPARSE_DATATYPE_FLOAT) {
    alpha_fill_random_s((float *)x_char, seed, len);
  } else if (dt == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    alpha_fill_random_d((double *)x_char, seed, len);
  } else if (dt == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    alpha_fill_random_c((ALPHA_Complex8 *)x_char, seed, len);
  } else if (dt == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    alpha_fill_random_z((ALPHA_Complex16 *)x_char, seed, len);
  }
}

void alpha_create_coo_wapper(matrix_data_t *matrix_data,
                             alphasparse_datatype_t data_type,
                             alphasparse_matrix_t *output) {
  int m = matrix_data->m;
  int k = matrix_data->k;
  int nnz = matrix_data->nnz;
  ALPHA_INT *row_index = matrix_data->row_index;
  ALPHA_INT *col_index = matrix_data->col_index;
  if (data_type != matrix_data->data_type) {
    printf("mismatched datatype of input argument and matrix\n");
    exit(-1);
  }
  if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    float *values = (float *)matrix_data->values;
    alpha_call_exit(
        alphasparse_s_create_coo(output, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                 nnz, row_index, col_index, values),
        "alphasparse_d_create_coo");
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    double *values = (double *)matrix_data->values;
    alpha_call_exit(
        alphasparse_d_create_coo(output, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                 nnz, row_index, col_index, values),
        "alphasparse_d_create_coo");
  } else if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    ALPHA_Complex8 *values = (ALPHA_Complex8 *)matrix_data->values;
    alpha_call_exit(
        alphasparse_c_create_coo(output, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                 nnz, row_index, col_index, values),
        "alphasparse_d_create_coo");
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    ALPHA_Complex16 *values = (ALPHA_Complex16 *)matrix_data->values;
    alpha_call_exit(
        alphasparse_z_create_coo(output, ALPHA_SPARSE_INDEX_BASE_ZERO, m, k,
                                 nnz, row_index, col_index, values),
        "alphasparse_d_create_coo");
  }
}

void alpha_convert_matrix_wapper(alphasparseFormat_t fmt,
                                 struct alpha_matrix_descr descr,
                                 alphasparse_layout_t layout,
                                 alphasparse_matrix_t input,
                                 alphasparse_matrix_t *output,
                                 int para_int1, int para_int2) {
  if (fmt == ALPHA_SPARSE_FORMAT_CSR) {
    alpha_call_exit(alphasparse_convert_csr(
                        input, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
                    "alphasparse_convert_csr");
  } else if (fmt == ALPHA_SPARSE_FORMAT_CSC) {
    alpha_call_exit(alphasparse_convert_csc(
                        input, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
                    "alphasparse_convert_csc");
  } else if (fmt == ALPHA_SPARSE_FORMAT_BSR) {
    alpha_call_exit(alphasparse_convert_bsr(
                        input, para_int1, layout,
                        ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
                    "alphasparse_convert_bsr");
  } else if (fmt == ALPHA_SPARSE_FORMAT_SKY) {
    alpha_call_exit(
        alphasparse_convert_sky(input, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE,
                                descr.mode, output),
        "alphasparse_convert_sky");
  } else if (fmt == ALPHA_SPARSE_FORMAT_DIA) {
    alpha_call_exit(alphasparse_convert_dia(
                        input, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
                    "alphasparse_convert_dia");
  } else if (fmt == ALPHA_SPARSE_FORMAT_ELL) {
    alpha_call_exit(alphasparse_convert_ell(
                        input, ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
                    "alphasparse_convert_ell");
  // } else if (fmt == ALPHA_SPARSE_FORMAT_GEBSR) {
  //   alpha_call_exit(
  //       alphasparse_convert_gebsr(input, para_int1, para_int2,
  //                                 ALPHA_SPARSE_LAYOUT_ROW_MAJOR,
  //                                 ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
  //       "alphasparse_convert_gebsr");
  // } else if (fmt == ALPHA_SPARSE_FORMAT_SELL_C_SIGMA) {
  //   alpha_call_exit(
  //       alphasparse_convert_sell_csigma(input, true, para_int1, para_int2,
  //                                 ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, output),
  //       "alphasparse_convert_gebsr");
  } else {
    printf("invalid format to conversion\n");
    exit(-1);
  }
}

void alpha_initialize_alpha_beta(char *alpha_char, char *beta_char,
                                 alphasparse_datatype_t data_type) {
  if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    float *alpha = (float *)alpha_char;
    float *beta = (float *)beta_char;
    *alpha = 2.f;
    *beta = 3.f;
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    double *alpha = (double *)alpha_char;
    double *beta = (double *)beta_char;
    *alpha = 2.f;
    *beta = 3.f;
  } else if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    ALPHA_Complex8 *alpha = (ALPHA_Complex8 *)alpha_char;
    ALPHA_Complex8 *beta = (ALPHA_Complex8 *)beta_char;
    alpha->real = 2.f;
    alpha->imag = 1.f;
    beta->real = 1.f;
    beta->imag = 2.f;
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    ALPHA_Complex16 *alpha = (ALPHA_Complex16 *)alpha_char;
    ALPHA_Complex16 *beta = (ALPHA_Complex16 *)beta_char;
    alpha->real = 2.;
    alpha->imag = 1.;
    beta->real = 1.;
    beta->imag = 2.;
  }
}
#ifdef __MKL__
// define this function since mkl cant convert coo2csc
static void alpha_convert_mkl_csc_d(alphasparse_datatype_t data_type,
                                    alphasparse_matrix_t src,
                                    sparse_matrix_t *dst) {
  if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    spmat_csc_s_t *mat = (spmat_csc_s_t *)src->mat;
    mkl_sparse_s_create_csc(dst, SPARSE_INDEX_BASE_ZERO, mat->rows, mat->cols,
                            mat->cols_start, mat->cols_end, mat->row_indx,
                            (float *)mat->values);
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    spmat_csc_d_t *mat = (spmat_csc_d_t *)src->mat;
    mkl_sparse_d_create_csc(dst, SPARSE_INDEX_BASE_ZERO, mat->rows, mat->cols,
                            mat->cols_start, mat->cols_end, mat->row_indx,
                            (double *)mat->values);
  } else if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    spmat_csc_c_t *mat = (spmat_csc_c_t *)src->mat;
    mkl_sparse_c_create_csc(dst, SPARSE_INDEX_BASE_ZERO, mat->rows, mat->cols,
                            mat->cols_start, mat->cols_end, mat->row_indx,
                            (MKL_Complex8 *)mat->values);
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    spmat_csc_z_t *mat = (spmat_csc_z_t *)src->mat;
    mkl_sparse_z_create_csc(dst, SPARSE_INDEX_BASE_ZERO, mat->rows, mat->cols,
                            mat->cols_start, mat->cols_end, mat->row_indx,
                            (MKL_Complex16 *)mat->values);
  } else {
    fprintf(stderr, "data type error\n");
    exit(-1);
  }
}
void mkl_create_coo_wapper(matrix_data_t *matrix_data,
                           alphasparse_datatype_t data_type,
                           sparse_matrix_t *output) {
  int m = matrix_data->m;
  int k = matrix_data->k;
  int nnz = matrix_data->nnz;
  MKL_INT *row_index = matrix_data->row_index;
  MKL_INT *col_index = matrix_data->col_index;
  if (data_type != matrix_data->data_type) {
    printf("mismatched datatype of input argument and matrix\n");
    exit(-1);
  }
  if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    float *values = (float *)matrix_data->values;
    mkl_call_exit(mkl_sparse_s_create_coo(output, SPARSE_INDEX_BASE_ZERO, m, k,
                                          nnz, row_index, col_index, values),
                  "mkl_sparse_d_create_coo");
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    double *values = (double *)matrix_data->values;
    mkl_call_exit(mkl_sparse_d_create_coo(output, SPARSE_INDEX_BASE_ZERO, m, k,
                                          nnz, row_index, col_index, values),
                  "mkl_sparse_d_create_coo");
  } else if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    MKL_Complex8 *values = (MKL_Complex8 *)matrix_data->values;
    mkl_call_exit(mkl_sparse_c_create_coo(output, SPARSE_INDEX_BASE_ZERO, m, k,
                                          nnz, row_index, col_index, values),
                  "mkl_sparse_d_create_coo");
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    MKL_Complex16 *values = (MKL_Complex16 *)matrix_data->values;
    mkl_call_exit(mkl_sparse_z_create_coo(output, SPARSE_INDEX_BASE_ZERO, m, k,
                                          nnz, row_index, col_index, values),
                  "mkl_sparse_d_create_coo");
  }
}

void mkl_convert_matrix_wapper(alphasparseFormat_t fmt,
                               struct matrix_descr descr, sparse_layout_t layout, sparse_matrix_t input,
                               sparse_matrix_t *output, int row_block,
                               int col_block) {
  if (fmt == ALPHA_SPARSE_FORMAT_CSR) {
    mkl_call_exit(
        mkl_sparse_convert_csr(input, SPARSE_OPERATION_NON_TRANSPOSE, output),
        "mkl_sparse_convert_csr");
  } else if (fmt == ALPHA_SPARSE_FORMAT_CSC) {
    mkl_call_exit(
      mkl_sparse_convert_csr(input, SPARSE_OPERATION_TRANSPOSE, output),
      "mkl_sparse_convert_csc");
    // fprintf(stderr, "fmt not supported!\n");
    // exit(-1);
  } else if (fmt == ALPHA_SPARSE_FORMAT_BSR) {
    mkl_call_exit(
        mkl_sparse_convert_bsr(input, row_block, layout,
                               SPARSE_OPERATION_NON_TRANSPOSE, output),
        "mkl_sparse_convert_bsr");
  } else {
    printf("invalid format to conversion\n");
    exit(-1);
  }
}

void mkl_initialize_alpha_beta(char *alpha_char, char *beta_char,
                               alphasparse_datatype_t data_type) {
  if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT) {
    float *alpha = (float *)alpha_char;
    float *beta = (float *)beta_char;
    *alpha = 2.f;
    *beta = 3.f;
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    double *alpha = (double *)alpha_char;
    double *beta = (double *)beta_char;
    *alpha = 2.f;
    *beta = 3.f;
  } else if (data_type == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    MKL_Complex8 *alpha = (MKL_Complex8 *)alpha_char;
    MKL_Complex8 *beta = (MKL_Complex8 *)beta_char;
    alpha->real = 2.f;
    alpha->imag = 1.f;
    beta->real = 1.f;
    beta->imag = 2.f;
  } else if (data_type == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    MKL_Complex16 *alpha = (MKL_Complex16 *)alpha_char;
    MKL_Complex16 *beta = (MKL_Complex16 *)beta_char;
    alpha->real = 2.;
    alpha->imag = 1.;
    beta->real = 1.;
    beta->imag = 2.;
  }
}

#endif

void check_arm(alphasparse_datatype_t dt, const char *icty_char, const int len,
               const char *icty_plain_char) {
  if (dt == ALPHA_SPARSE_DATATYPE_FLOAT) {
    check_s((float *)icty_char, len, (float *)icty_plain_char, len);
  } else if (dt == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    check_d((double *)icty_char, len, (double *)icty_plain_char, len);
  } else if (dt == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    check_c((ALPHA_Complex8 *)icty_char, len, (ALPHA_Complex8 *)icty_plain_char,
            len);
  } else if (dt == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    check_z((ALPHA_Complex16 *)icty_char, len,
            (ALPHA_Complex16 *)icty_plain_char, len);
  }
}
void destory_matrix_data(matrix_data_t *matrix_data) {
  alpha_free(matrix_data->row_index);
  alpha_free(matrix_data->col_index);
  alpha_free(matrix_data->values);
}