#include "include/args.h"

#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #ifndef __MKL__
// #define __MKL__
// #endif  

char *stort_options = "";
struct option long_options[] = {
    {"help", no_argument, NULL, 1},
    {"check", no_argument, NULL, 2},
    {"thread-num", optional_argument, NULL, 3},

    {"transA", optional_argument, NULL, 4},
    {"transB", optional_argument, NULL, 5},

    {"typeA", optional_argument, NULL, 6},
    {"fillA", optional_argument, NULL, 7},
    {"diagA", optional_argument, NULL, 8},

    {"typeB", optional_argument, NULL, 9},
    {"fillB", optional_argument, NULL, 10},
    {"diagB", optional_argument, NULL, 11},

    {"data-file", optional_argument, NULL, 12},
    {"data-fileA", optional_argument, NULL, 13},
    {"data-fileB", optional_argument, NULL, 14},
    {"data-fileC", optional_argument, NULL, 24},  // 24here

    {"layout", optional_argument, NULL, 15},
    {"layoutB", optional_argument, NULL, 16},
    {"layoutC", optional_argument, NULL, 17},

    {"columns", optional_argument, NULL, 18},

    {"iter", optional_argument, NULL, 19},

    {"format", optional_argument, NULL, 24},
    {"formatA", optional_argument, NULL, 20},
    {"formatB", optional_argument, NULL, 21},

    {"data-type", optional_argument, NULL, 22},
    {"warm", no_argument, NULL, 23},
    {"param0", optional_argument, NULL, 25},
    {"param1", optional_argument, NULL, 26},
    {"param2", optional_argument, NULL, 27},
};

void print_help() {
  printf("--help\n\thelp\n\n");
  printf("--check\n\tcheck or not,default:no check\n\n");
  printf("--warm\n\twarming up or not,default:no warming up\n\n");
  printf("--thread-num=<int>\n\tthread number,default:%d\n\n", DEFAULT_THREAD_NUM);
  printf("--data-file=<file>\n\tdata file");
  printf("--data-fileA=<file>\n\tdata file");
  printf("--data-fileB=<file>\n\tdata file");
  printf("--data-fileC=<file>\n\tdata file");
  printf("--columns=<int>\n\tdense matrix columns,default:equal to k\n\n");
  printf("--iter=<int>\n\tnumber of test iterations,default:1\n\n");
  printf("--layout=<R|C>\n\tdense matrix layout,default:R\n\n");
  printf("--transA=<N|T|H>\n\tmatrix A operation,default:N\n\n");
  printf("--transB=<N|T|H>\n\tmatrix B operation,default:N\n\n");
  printf("--typeA=<G|S|H|T|D|BT|BD>\n\tmatrix A type,default:G\n\n");
  printf("--fillA=<L|U>\n\tmatrix A fill mode,default:L\n\n");
  printf("--diagA=<N|U>\n\tmatrix A diag type,default:N\n\n");
  printf("--typeB=<G|S|H|T|D|BT|BD>\n\tmatrix A type,default:G\n\n");
  printf("--fillB=<L|U>\n\tmatrix A fill mode,default:L\n\n");
  printf("--diagB=<N|U>\n\tmatrix A diag type,default:N\n\n");
  printf(
      "--formatA=<csr|coo|csc|(ge)bsr|dia|sky|ell|hyb>\n\tmatrix A format to create,default:csr\n\n");
  printf(
      "--formatB=<csr|coo|csc|(ge)bsr|dia|sky|ell|hyb>\n\tmatrix B format to create,default:consistent "
      "with formatA\n\n");
  printf("--data-type=<S|D|C|Z>\n\tdata-type to create,default:S\n\n");
  exit(-1);
}

void args_help(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 1) print_help();
}

bool args_get_if_check(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 2) {
      return true;
    }
  return DEFAULT_CHECK;
}

int args_get_thread_num(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 3) {
      return atoi(optarg);
    }
  return DEFAULT_THREAD_NUM;
}

int args_get_columns(const int argc, const char *argv[], int k) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 18) {
      return atoi(optarg);
    }
  return k;
}

int args_get_iter(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 19) {
      return atoi(optarg);
    }
  return DEFAULT_ITER;
}
int args_get_param0(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 25) {
      return atoi(optarg);
    }
  return DEFAULT_PARAM;
}

int args_get_param1(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 26) {
      return atoi(optarg);
    }
  return DEFAULT_PARAM;
}

bool args_get_if_warm(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 23) {
      return true;
    }
  return DEFAULT_WARM;
}

const char *args_get_data_file(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 12) {
      return optarg;
    }
  return DEFAULT_DATA_FILE;
}

const char *args_get_data_fileA(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  char *data_file = NULL;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 13) {
      return optarg;
    } else if (opt == 12) {
      data_file = optarg;
    }
  const char *ret_data_file = data_file == NULL ? DEFAULT_DATA_FILE : data_file;
  return ret_data_file;
}

const char *args_get_data_fileB(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  char *data_file = NULL;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 14) {
      return optarg;
    } else if (opt == 12) {
      data_file = optarg;
    }
  const char *ret_data_file = data_file == NULL ? DEFAULT_DATA_FILE : data_file;
  return ret_data_file;
}

const char *args_get_data_fileC(const int argc, const char *argv[]) {
  optind = 0;
  int opt;
  int option_index;
  char *data_file = NULL;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == 24) {
      return optarg;
    } else if (opt == 12) {
      data_file = optarg;
    }
  const char *ret_data_file = data_file == NULL ? DEFAULT_DATA_FILE : data_file;
  return ret_data_file;
}

alphasparse_layout_t alpha_args_get_layout(const int argc, const char *argv[]) {
  return alpha_args_get_layout_helper(argc, argv, 15);
}

alphasparse_layout_t alpha_args_get_layoutA(const int argc, const char *argv[]) {
  return alpha_args_get_layout_helper(argc, argv, 16);
}

alphasparse_layout_t alpha_args_get_layoutB(const int argc, const char *argv[]) {
  return alpha_args_get_layout_helper(argc, argv, 17);
}
alphasparseFormat_t alpha_args_get_format(const int argc, const char *argv[]) {
  return alpha_args_get_format_helper(argc, argv, 24);
}
alphasparseFormat_t alpha_args_get_formatA(const int argc, const char *argv[]) {
  return alpha_args_get_format_helper(argc, argv, 20);
}
alphasparseFormat_t alpha_args_get_formatB(const int argc, const char *argv[]) {
  return alpha_args_get_format_helper(argc, argv, 21);
}
alphasparse_datatype_t alpha_args_get_data_type(const int argc, const char *argv[]) {
  return alpha_args_get_data_type_helper(argc, argv, 22);
}
struct alpha_matrix_descr alpha_args_get_matrix_descrA(const int argc, const char *argv[]) {
  return alpha_args_get_matrix_descr_helper(argc, argv, 6, 7, 8);
}

struct alpha_matrix_descr alpha_args_get_matrix_descrB(const int argc, const char *argv[]) {
  return alpha_args_get_matrix_descr_helper(argc, argv, 9, 10, 11);
}

alphasparseOperation_t alpha_args_get_transA(const int argc, const char *argv[]) {
  return alpha_args_get_trans_helper(argc, argv, 4);
}

alphasparseOperation_t alpha_args_get_transB(const int argc, const char *argv[]) {
  return alpha_args_get_trans_helper(argc, argv, 5);
}

alphasparse_layout_t alpha_args_get_layout_helper(const int argc, const char *argv[],
                                               const int layout_opt) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == layout_opt) {
      return alphasparse_layout_parse(optarg);
    }
  return DEFAULT_LAYOUT;
}

alphasparseFormat_t alpha_args_get_format_helper(const int argc, const char *argv[], int format_opt) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == format_opt) {
      return alphasparse_format_parse(optarg);
    }
  return DEFAULT_FORMAT;
}
alphasparse_datatype_t alpha_args_get_data_type_helper(const int argc, const char *argv[],
                                                    int format_opt) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == format_opt) {
      return alphasparse_data_type_parse(optarg);
    }
  return DEFAULT_DATA_TYPE;
}
struct alpha_matrix_descr alpha_args_get_matrix_descr_helper(const int argc, const char *argv[],
                                                         const int type_opt, const int fill_opt,
                                                         const int diag_opt) {
  optind = 0;
  int opt;
  int option_index;
  struct alpha_matrix_descr ret = {.type = ALPHA_SPARSE_MATRIX_TYPE_GENERAL};
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == type_opt)
      ret.type = alphasparse_matrix_type_parse(optarg);
    else if (opt == fill_opt)
      ret.mode = alphasparse_fill_mode_parse(optarg);
    else if (opt == diag_opt)
      ret.diag = alphasparse_diag_type_parse(optarg);
  return ret;
}

alphasparseOperation_t alpha_args_get_trans_helper(const int argc, const char *argv[],
                                                 const int trans_opt) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == trans_opt) return alphasparse_operation_parse(optarg);
  return DEFAULT_SPARSE_OPERATION;
}

alphasparse_layout_t alphasparse_layout_parse(const char *arg) {
  if (strcmp("R", arg) == 0 || strcmp("ROW", arg) == 0) return ALPHA_SPARSE_LAYOUT_ROW_MAJOR;
  if (strcmp("C", arg) == 0 || strcmp("COL", arg) == 0 || strcmp("COLUMN", arg) == 0)
    return ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR;
  printf("invalid layout value!!! %s\n", arg);
  exit(-1);
}

alphasparseFormat_t alphasparse_format_parse(const char *arg) {
  if (strcmp("coo", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_COO;
  }
  if (strcmp("csr", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_CSR;
  }
  if (strcmp("csc", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_CSC;
  }
  if (strcmp("bsr", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_BSR;
  }
  if (strcmp("sky", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_SKY;
  }
  if (strcmp("dia", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_DIA;
  }
  if (strcmp("ell", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_ELL;
  }
  if (strcmp("hyb", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_ELL;
  }
  if (strcmp("gebsr", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_GEBSR;
  }
  if (strcmp("cooaos", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_COOAOS;
  }
  if (strcmp("csr5", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_CSR5;
  }
  if (strcmp("sell_c_sigma", arg) == 0) {
    return ALPHA_SPARSE_FORMAT_SELL_C_SIGMA;
  }
  printf("invalid format %s\n", arg);
  exit(-1);
}
alphasparse_datatype_t alphasparse_data_type_parse(const char *arg) {
  if (strcmp("S", arg) == 0) {
    return ALPHA_SPARSE_DATATYPE_FLOAT;
  }
  if (strcmp("D", arg) == 0) {
    return ALPHA_SPARSE_DATATYPE_DOUBLE;
  }
  if (strcmp("C", arg) == 0) {
    return ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX;
  }
  if (strcmp("Z", arg) == 0) {
    return ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX;
  }
  printf("invalid data_type %s\n", arg);
  exit(-1);
}
alphasparseOperation_t alphasparse_operation_parse(const char *arg) {
  if (strcmp("N", arg) == 0 || strcmp("NON_TRANSPOSE", arg) == 0)
    return ALPHA_SPARSE_OPERATION_NON_TRANSPOSE;
  if (strcmp("T", arg) == 0 || strcmp("TRANSPOSE", arg) == 0) return ALPHA_SPARSE_OPERATION_TRANSPOSE;
  if (strcmp("H", arg) == 0 || strcmp("CONJUGATE_TRANSPOSE", arg) == 0)
    return ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  printf("invalid operation A value!!! %s\n", arg);
  exit(-1);
}

alphasparse_matrix_type_t alphasparse_matrix_type_parse(const char *arg) {
  if (strcmp("G", arg) == 0 || strcmp("GENERAL", arg) == 0) return ALPHA_SPARSE_MATRIX_TYPE_GENERAL;
  if (strcmp("S", arg) == 0 || strcmp("SYMMETRIC", arg) == 0)
    return ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC;
  if (strcmp("H", arg) == 0 || strcmp("HERMITIAN", arg) == 0)
    return ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN;
  if (strcmp("T", arg) == 0 || strcmp("TRIANGULAR", arg) == 0)
    return ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR;
  if (strcmp("D", arg) == 0 || strcmp("DIAGONAL", arg) == 0) return ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL;
  if (strcmp("BT", arg) == 0 || strcmp("BLOCK_TRIANGULAR", arg) == 0)
    return ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR;
  if (strcmp("BD", arg) == 0 || strcmp("BLOCK_DIAGONAL", arg) == 0)
    return ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL;
  printf("invalid type A value!!! %s\n", arg);
  exit(-1);
}

alphasparse_fill_mode_t alphasparse_fill_mode_parse(const char *arg) {
  if (strcmp("L", arg) == 0 || strcmp("LOWER", arg) == 0) return ALPHA_SPARSE_FILL_MODE_LOWER;
  if (strcmp("U", arg) == 0 || strcmp("UPPER", arg) == 0) return ALPHA_SPARSE_FILL_MODE_UPPER;
  printf("invalid fill A value!!! %s\n", arg);
  exit(-1);
}

alphasparse_diag_type_t alphasparse_diag_type_parse(const char *arg) {
  if (strcmp("N", optarg) == 0 || strcmp("NON_UNIT", optarg) == 0) return ALPHA_SPARSE_DIAG_NON_UNIT;
  if (strcmp("U", optarg) == 0 || strcmp("UNIT", optarg) == 0) return ALPHA_SPARSE_DIAG_UNIT;
  printf("invalid diag A value!!! %s\n", optarg);
  exit(-1);
}

#ifdef __MKL__

sparse_layout_t mkl_args_get_layout(const int argc, const char *argv[]) {
  return mkl_args_get_layout_helper(argc, argv, 15);
}

sparse_layout_t mkl_args_get_layoutA(const int argc, const char *argv[]) {
  return mkl_args_get_layout_helper(argc, argv, 16);
}
 
sparse_layout_t mkl_args_get_layoutB(const int argc, const char *argv[]) {
  return mkl_args_get_layout_helper(argc, argv, 17);
}

struct matrix_descr mkl_args_get_matrix_descrA(const int argc, const char *argv[]) {
  return mkl_args_get_matrix_descr_helper(argc, argv, 6, 7, 8);
}

struct matrix_descr mkl_args_get_matrix_descrB(const int argc, const char *argv[]) {
  return mkl_args_get_matrix_descr_helper(argc, argv, 9, 10, 11);
}

sparse_operation_t mkl_args_get_transA(const int argc, const char *argv[]) {
  return mkl_args_get_trans_helper(argc, argv, 4);
}

sparse_operation_t mkl_args_get_transB(const int argc, const char *argv[]) {
  return mkl_args_get_trans_helper(argc, argv, 5);
}

sparse_layout_t mkl_args_get_layout_helper(const int argc, const char *argv[],
                                           const int layout_opt) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == layout_opt) {
      return mkl_sparse_layout_parse(optarg);
    }
  return SPARSE_LAYOUT_ROW_MAJOR;
}

struct matrix_descr mkl_args_get_matrix_descr_helper(const int argc, const char *argv[],
                                                     const int type_opt, const int fill_opt,
                                                     const int diag_opt) {
  optind = 0;
  int opt;
  int option_index;
  struct matrix_descr ret = {.type = SPARSE_MATRIX_TYPE_GENERAL};
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == type_opt)
      ret.type = mkl_sparse_matrix_type_parse(optarg);
    else if (opt == fill_opt)
      ret.mode = mkl_sparse_fill_mode_parse(optarg);
    else if (opt == diag_opt)
      ret.diag = mkl_sparse_diag_type_parse(optarg);
  return ret;
}

sparse_operation_t mkl_args_get_trans_helper(const int argc, const char *argv[],
                                             const int trans_opt) {
  optind = 0;
  int opt;
  int option_index;
  while ((opt = getopt_long_only(argc, (char *const *)argv, stort_options, long_options,
                                 &option_index)) != -1)
    if (opt == trans_opt) return mkl_sparse_operation_parse(optarg);
  return SPARSE_OPERATION_NON_TRANSPOSE;
}

sparse_layout_t mkl_sparse_layout_parse(const char *arg) {
  if (strcmp("R", arg) == 0 || strcmp("ROW", arg) == 0) return SPARSE_LAYOUT_ROW_MAJOR;
  if (strcmp("C", arg) == 0 || strcmp("COL", arg) == 0 || strcmp("COLUMN", arg) == 0)
    return SPARSE_LAYOUT_COLUMN_MAJOR;
  printf("invalid layout value!!! %s\n", arg);
  exit(-1);
}

sparse_operation_t mkl_sparse_operation_parse(const char *arg) {
  if (strcmp("N", arg) == 0 || strcmp("NON_TRANSPOSE", arg) == 0)
    return SPARSE_OPERATION_NON_TRANSPOSE;
  if (strcmp("T", arg) == 0 || strcmp("TRANSPOSE", arg) == 0) return SPARSE_OPERATION_TRANSPOSE;
  if (strcmp("H", arg) == 0 || strcmp("CONJUGATE_TRANSPOSE", arg) == 0)
    return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  printf("invalid operation A value!!! %s\n", arg);
  exit(-1);
}

sparse_matrix_type_t mkl_sparse_matrix_type_parse(const char *arg) {
  if (strcmp("G", arg) == 0 || strcmp("GENERAL", arg) == 0) return SPARSE_MATRIX_TYPE_GENERAL;
  if (strcmp("S", arg) == 0 || strcmp("SYMMETRIC", arg) == 0) return SPARSE_MATRIX_TYPE_SYMMETRIC;
  if (strcmp("H", arg) == 0 || strcmp("HERMITIAN", arg) == 0) return SPARSE_MATRIX_TYPE_HERMITIAN;
  if (strcmp("T", arg) == 0 || strcmp("TRIANGULAR", arg) == 0) return SPARSE_MATRIX_TYPE_TRIANGULAR;
  if (strcmp("D", arg) == 0 || strcmp("DIAGONAL", arg) == 0) return SPARSE_MATRIX_TYPE_DIAGONAL;
  if (strcmp("BT", arg) == 0 || strcmp("BLOCK_TRIANGULAR", arg) == 0)
    return SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR;
  if (strcmp("BD", arg) == 0 || strcmp("BLOCK_DIAGONAL", arg) == 0)
    return SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL;
  printf("invalid type A value!!! %s\n", arg);
  exit(-1);
}

sparse_fill_mode_t mkl_sparse_fill_mode_parse(const char *arg) {
  if (strcmp("L", arg) == 0 || strcmp("LOWER", arg) == 0) return SPARSE_FILL_MODE_LOWER;
  if (strcmp("U", arg) == 0 || strcmp("UPPER", arg) == 0) return SPARSE_FILL_MODE_UPPER;
  printf("invalid fill A value!!! %s\n", arg);
  exit(-1);
}

sparse_diag_type_t mkl_sparse_diag_type_parse(const char *arg) {
  if (strcmp("N", optarg) == 0 || strcmp("NON_UNIT", optarg) == 0) return SPARSE_DIAG_NON_UNIT;
  if (strcmp("U", optarg) == 0 || strcmp("UNIT", optarg) == 0) return SPARSE_DIAG_UNIT;
  printf("invalid diag A value!!! %s\n", optarg);
  exit(-1);
}
#endif
