#pragma once
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief header for basic types and constants for ict spblas API
 */

/* status of the routines */
typedef enum {
  ALPHA_SPARSE_STATUS_SUCCESS = 0,          /* the operation was successful */
  ALPHA_SPARSE_STATUS_NOT_INITIALIZED = 1,  /* empty handle or matrix arrays */
  ALPHA_SPARSE_STATUS_ALLOC_FAILED = 2,     /* internal error: memory allocation failed */
  ALPHA_SPARSE_STATUS_INVALID_VALUE = 3,    /* invalid input value */
  ALPHA_SPARSE_STATUS_EXECUTION_FAILED = 4, /* e.g. 0-diagonal element for triangular solver, etc. */
  ALPHA_SPARSE_STATUS_INTERNAL_ERROR = 5,   /* internal error */
  ALPHA_SPARSE_STATUS_NOT_SUPPORTED = 6,    /* e.g. operation for double precision doesn't support other types */
  ALPHA_SPARSE_STATUS_INVALID_POINTER = 7,  /* e.g. invlaid pointers */
  ALPHA_SPARSE_STATUS_INVALID_HANDLE = 8,   /* e.g. invlaid handle */
  ALPHA_SPARSE_STATUS_INVALID_KERNEL_PARAM = 9, /* e.g. invalid param for sell-c-sigma gemv kernel */
  ALPHA_SPARSE_STATUS_INVALID_SIZE = 10
} alphasparseStatus_t;

/* sparse matrix operations */
typedef enum {
  ALPHA_SPARSE_OPERATION_NON_TRANSPOSE = 0,
  ALPHA_SPARSE_OPERATION_TRANSPOSE = 1,
  ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} alphasparseOperation_t;

#define ALPHA_SPARSE_OPERATION_NUM 3

/* supported matrix types */
typedef enum {
  ALPHA_SPARSE_MATRIX_TYPE_GENERAL = 0,   /*    General case                    */
  ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC = 1, /*    Triangular part of              */
  ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN = 2, /*    the matrix is to be processed   */
  ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR = 3,
  ALPHA_SPARSE_MATRIX_TYPE_DIAGONAL =
      4, /* diagonal matrix; only diagonal elements will be processed */
  ALPHA_SPARSE_MATRIX_TYPE_BLOCK_TRIANGULAR = 5,
  ALPHA_SPARSE_MATRIX_TYPE_BLOCK_DIAGONAL =
      6 /* block-diagonal matrix; only diagonal blocks will be processed */
} alphasparse_matrix_type_t;

#define ALPHA_SPARSE_MATRIX_TYPE_NUM 2

/* sparse matrix indexing: C-style or Fortran-style */
typedef enum {
  ALPHA_SPARSE_INDEX_BASE_ZERO = 0, /* C-style */
  ALPHA_SPARSE_INDEX_BASE_ONE = 1   /* Fortran-style */
} alphasparseIndexBase_t;

#define ALPHA_SPARSE_INDEX_NUM 2

/* applies to triangular matrices only ( ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC,
 * ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN, ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR ) */
typedef enum {
  ALPHA_SPARSE_FILL_MODE_LOWER = 0, /* lower triangular part of the matrix is stored */
  ALPHA_SPARSE_FILL_MODE_UPPER = 1, /* upper triangular part of the matrix is stored */
} alphasparse_fill_mode_t;

#define ALPHA_SPARSE_FILL_MODE_NUM 2

/* applies to triangular matrices only ( ALPHA_SPARSE_MATRIX_TYPE_SYMMETRIC,
 * ALPHA_SPARSE_MATRIX_TYPE_HERMITIAN, ALPHA_SPARSE_MATRIX_TYPE_TRIANGULAR ) */
typedef enum {
  ALPHA_SPARSE_DIAG_NON_UNIT = 0, /* triangular matrix with non-unit diagonal */
  ALPHA_SPARSE_DIAG_UNIT = 1      /* triangular matrix with unit diagonal */
} alphasparse_diag_type_t;

#define ALPHA_SPARSE_DIAG_TYPE_NUM 2

/* applicable for Level 3 operations with dense matrices; describes storage scheme for dense matrix
 * (row major or column major) */
typedef enum {
  ALPHA_SPARSE_LAYOUT_ROW_MAJOR = 0,   /* C-style */
  ALPHA_SPARSE_LAYOUT_COLUMN_MAJOR = 1 /* Fortran-style */
} alphasparse_layout_t;

/* applicable for Level 3 operations with dense matrices; describes storage scheme for dense matrix
 * (row major or column major) */
typedef enum {
  ALPHASPARSE_ORDER_ROW = 0,   /* C-style */
  ALPHASPARSE_ORDER_COL = 1 /* Fortran-style */
} alphasparseOrder_t;

#define ALPHA_SPARSE_LAYOUT_NUM 2

/* verbose mode; if verbose mode activated, handle should collect and report profiling /
 * optimization info */
typedef enum {
  ALPHA_SPARSE_VERBOSE_OFF = 0,
  ALPHA_SPARSE_VERBOSE_BASIC =
      1, /* output contains high-level information about optimization algorithms, issues, etc. */
  ALPHA_SPARSE_VERBOSE_EXTENDED = 2 /* provide detailed output information */
} alpha_verbose_mode_t;

/* memory optimization hints from user: describe how much memory could be used on optimization stage
 */
typedef enum {
  ALPHA_SPARSE_MEMORY_NONE =
      0, /* no memory should be allocated for matrix values and structures; auxiliary structures
            could be created only for workload balancing, parallelization, etc. */
  ALPHA_SPARSE_MEMORY_AGGRESSIVE = 1 /* matrix could be converted to any internal format */
} alphasparse_memory_usage_t;

typedef enum {
  ALPHA_SPARSE_STAGE_FULL_MULT = 0,
  ALPHA_SPARSE_STAGE_NNZ_COUNT = 1,
  ALPHA_SPARSE_STAGE_FINALIZE_MULT = 2,
  ALPHA_SPARSE_STAGE_FULL_MULT_NO_VAL = 3,
  ALPHA_SPARSE_STAGE_FINALIZE_MULT_NO_VAL = 4
} alphasparse_request_t;

/*************************************************************************************************/
/*** Opaque structure for sparse matrix in internal format, further D - means double precision ***/
/*************************************************************************************************/

/*
 * ict sparse matirx implement;
 * ----------------------------------------------------------------------------------------------------------------------
 */

typedef enum {
  ALPHA_SPARSE_FORMAT_COO = 0,
  ALPHA_SPARSE_FORMAT_CSR = 1,
  ALPHA_SPARSE_FORMAT_CSC = 2,
  ALPHA_SPARSE_FORMAT_BSR = 3,
  ALPHA_SPARSE_FORMAT_SKY = 4,
  ALPHA_SPARSE_FORMAT_DIA = 5,
  ALPHA_SPARSE_FORMAT_ELL = 6,
  ALPHA_SPARSE_FORMAT_GEBSR = 7,
  ALPHA_SPARSE_FORMAT_HYB = 8,
  ALPHA_SPARSE_FORMAT_COOAOS = 9,
  ALPHA_SPARSE_FORMAT_CSR5 = 10,
  ALPHA_SPARSE_FORMAT_SELL_C_SIGMA = 11,
  ALPHA_SPARSE_FORMAT_ELLR = 12,
  ALPHA_SPARSE_FORMAT_BLOCKED_ELL = 13
} alphasparseFormat_t;

#define ALPHA_SPARSE_FORMAT_NUM 13

typedef enum {
  ALPHA_R_32F = 0,
  ALPHA_R_64F = 1,
  ALPHA_C_32F = 2,
  ALPHA_C_64F = 3,
  ALPHA_R_16F = 4,
  ALPHA_R_16BF = 5,
  ALPHA_C_16F = 6,
  ALPHA_C_16BF = 7,
  ALPHA_R_8I = 8,
  ALPHA_R_32I = 9
} alphasparseDataType;

#define ALPHA_SPARSE_DATATYPE_NUM 9

// typedef enum {
//   ALPHA_SPARSE_FORMAT_COO = 0,
//   ALPHA_SPARSE_FORMAT_CSR = 1,
//   ALPHA_SPARSE_FORMAT_CSC = 2,
//   ALPHA_SPARSE_FORMAT_BSR = 3,
//   ALPHA_SPARSE_FORMAT_SKY = 4,
//   ALPHA_SPARSE_FORMAT_DIA = 5,
//   ALPHA_SPARSE_FORMAT_ELL = 6,
//   ALPHA_SPARSE_FORMAT_GEBSR = 7,
//   ALPHA_SPARSE_FORMAT_HYB = 8,
//   ALPHA_SPARSE_FORMAT_COOAOS = 9,
//   ALPHA_SPARSE_FORMAT_CSR5 = 10,
//   ALPHA_SPARSE_FORMAT_SELL_C_SIGMA = 11,
//   ALPHA_SPARSE_FORMAT_ELLR = 12
// } alphasparseFormat_t;

typedef enum {
  ALPHA_SPARSE_DATATYPE_FLOAT = 0,
  ALPHA_SPARSE_DATATYPE_DOUBLE = 1,
  ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX = 2,
  ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX = 3,
  ALPHA_SPARSE_DATATYPE_HALF_FLOAT = 4,
  ALPHA_SPARSE_DATATYPE_HALF_DOUBLE = 5,
  ALPHA_SPARSE_DATATYPE_HALF_FLOAT_COMPLEX = 6,
  ALPHA_SPARSE_DATATYPE_HALF_DOUBLE_COMPLEX = 7
} alphasparse_datatype_t;

#define ALPHA_SPARSE_DATATYPE_NUM_OLD 8

#ifndef COMPLEX
#ifndef DOUBLE
#ifndef HALF
#define ALPHA_SPARSE_DATATYPE ALPHA_R_32F
#else
#define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_HALF
#endif
#else
#define ALPHA_SPARSE_DATATYPE ALPHA_R_64F
#endif
#else
#ifndef DOUBLE
#define ALPHA_SPARSE_DATATYPE ALPHA_C_32F
#else
#define ALPHA_SPARSE_DATATYPE ALPHA_C_64F
#endif
#endif

#include "types.h"
struct _internal_spmat
{
  mutable bool analysed{};

  int64_t rows{};
  int64_t cols{};
  int64_t nnz{};
  int64_t ndiag{};
  int64_t lval{};

  ALPHA_INT* row_data{};
  ALPHA_INT* col_data{};
  ALPHA_INT* ind_data{};
  ALPHA_INT* dis_data{};
  ALPHA_INT* pointers{};
  ALPHA_INT* reoeders{};
  void* val_data{};

  const int* const_row_data{};
  const int* const_col_data{};
  const int* const_ind_data{};
  const void* const_val_data{};

  alphasparseIndexBase_t idx_base{};
  alphasparse_fill_mode_t fill{};

  int64_t block_dim{};
  int64_t row_block_dim{};
  int64_t col_block_dim{};
  int64_t block_layout{};
  
  int64_t ell_cols{};
  int64_t ell_width{};

  int64_t sell_C{};
  int64_t sell_sigma{};
  int64_t sell_blocks{};
  int64_t sell_nnz{};

  int64_t batch_count{};
  int64_t batch_stride{};
  int64_t offsets_batch_stride{};
  int64_t columns_values_batch_stride{};
  bool ordered;
};

typedef _internal_spmat *internal_spmat;
// typedef void *internal_spmat;
typedef void *alpha_internal_vector;
struct alphasparse_inspector__;
#ifdef __cplusplus
typedef alphasparse_inspector__ *alphasparse_inspector_t;
#endif
typedef struct alphasparse_inspector__ *alphasparse_inspector_t;

typedef struct {
  internal_spmat mat;
  alphasparseFormat_t format;      // csr,coo,csc,bsr,ell,dia,sky...
  alphasparseDataType datatype;  // s,d,c,z
} alphasparse_matrix_base;

typedef alphasparse_matrix_base *alphasparse_matrix_base_t;

typedef struct {
  internal_spmat mat;
  alphasparseFormat_t format;        // csr,coo,csc,bsr,ell,dia,sky...
  alphasparseDataType datatype;    // s,d,c,z,h
  void *info;                     // for autotuning, alphasparse_mat_info_t
  // internal_spmat mat;
  alphasparse_datatype_t datatype_cpu;    // s,d,c,z
  alphasparse_inspector_t inspector;  // for autotuning
  void *dcu_info;                     // for dcu autotuning, alphasparse_dcu_mat_info_t
} alphasparse_matrix;

typedef alphasparse_matrix *alphasparse_matrix_t;

/*
 * ----------------------------------------------------------------------------------------------------------------------
 */

/*structures for usages*/
struct alpha_matrix_descr {
  alphasparse_matrix_type_t
      type; /* matrix type: general, diagonal or triangular / symmetric / hermitian */
  alphasparse_fill_mode_t mode; /* upper or lower triangular part of the matrix ( for triangular /
                                  symmetric / hermitian case) */
  alphasparse_diag_type_t
      diag; /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
  alphasparseIndexBase_t base; /* C-style or Fortran-style*/
};

typedef struct alpha_matrix_descr *alpha_matrix_descr_t;
typedef enum {
  ALPHA_SPARSE_POINTER_MODE_HOST = 0,
  ALPHA_SPARSE_POINTER_MODE_DVICE = 1,
} alphasparse_pointer_mode_t;

typedef enum {
  ALPHA_SPARSE_LAYER_MODE_NONE = 0,      /**< layer is not active. */
  ALPHA_SPARSE_LAYER_MODE_LOG_TRACE = 1, /**< layer is in logging mode. */
  ALPHA_SPARSE_LAYER_MODE_LOG_BENCH = 2  /**< layer is in benchmarking mode. */
} alphasparse_layer_mode_t;

typedef enum {
  ALPHA_SPARSE_INDEXTYPE_U16 = 1, /**< 16 bit unsigned integer. */
  ALPHA_SPARSE_INDEXTYPE_I32 = 2, /**< 32 bit signed integer. */
  ALPHA_SPARSE_INDEXTYPE_I64 = 3  /**< 64 bit signed integer. */
} alphasparseIndexType_t;

typedef enum {
  ALPHA_SPARSE_HYB_PARTITION_AUTO = 0, /**< automatically decide on ELL nnz per row. */
  ALPHA_SPARSE_HYB_PARTITION_USER = 1, /**< user given ELL nnz per row. */
  ALPHA_SPARSE_HYB_PARTITION_MAX = 2   /**< max ELL nnz per row, no COO part. */
} alphasparse_hyb_partition_t;
typedef enum {
  ALPHA_SPARSE_ANALYSIS_POLICY_REUSE = 0, /**< try to re-use meta data. */
  ALPHA_SPARSE_ANALYSIS_POLICY_FORCE = 1  /**< force to re-build meta data. */
} alphasparse_analysis_policy_t;

typedef enum {
  ALPHA_SPARSE_SOLVE_POLICY_AUTO = 0 /**< automatically decide on level information. */
} alphasparse_solve_policy_t;

typedef enum {
  ALPHA_SPARSE_ACTION_SYMBOLIC = 0, /**< Operate only on indices. */
  ALPHA_SPARSE_ACTION_NUMERIC = 1   /**< Operate on data and indices. */
} alphasparse_action_t;

typedef enum {
  ALPHA_SPARSE_DENSE_TO_SPARSE_ALG_DEFAULT =
      0, /**< Default dense to sparse algorithm for the given format. */
} alphasparse_dense_to_sparse_alg_t;

typedef enum {
  ALPHA_SPARSE_SPARSE_TO_DENSE_ALG_DEFAULT =
      0, /**< Default sparse to dense algorithm for the given format. */
} alphasparse_sparse_to_dense_alg_t;

typedef enum {
  ALPHA_SPARSE_SPMV_ALG_DEFAULT = 0, /**< Default SpMV algorithm for the given format. */
  ALPHA_SPARSE_SPMV_ALG_COO = 1,     /**< COO SpMV algorithm for COO matrices. */
  ALPHA_SPARSE_SPMV_ALG_CSR_ADAPTIVE = 2,   /**< CSR SpMV algorithm 1 (adaptive) for CSR matrices. */
  ALPHA_SPARSE_SPMV_ALG_CSR_STREAM = 3, /**< CSR SpMV algorithm 2 (stream) for CSR matrices. */
  ALPHA_SPARSE_SPMV_ALG_ELL = 4,         /**< ELL SpMV algorithm for ELL matrices. */
  ALPHA_SPARSE_SPMV_ALG_SCALAR = 5,
  ALPHA_SPARSE_SPMV_ALG_VECTOR = 6,
  ALPHA_SPARSE_SPMV_ROW_PARTITION = 7,
  ALPHA_SPARSE_SPMV_ADAPTIVE = 8,
  ALPHA_SPARSE_SPMV_ALG_MERGE = 9,
  ALPHA_SPARSE_SPMV_ALG_LINE = 10,
  ALPHA_SPARSE_SPMV_ALG_FLAT1 = 11,
  ALPHA_SPARSE_SPMV_ALG_FLAT4 = 12,
  ALPHA_SPARSE_SPMV_ALG_FLAT8 = 13,
  ALPHA_SPARSE_SPMV_ALG_ACC = 14,
  ALPHA_SPARSE_SPMV_ALG_FLAT = 15,
  ALPHA_SPARSE_SPMV_ALG_LINE_ENHANCE = 16,
  ALPHA_SPARSE_SPMV_ALG_LOAD_BALANCE = 17,
  ALPHA_SPARSE_SPMV_ALG_PFLAT = 18,
} alphasparseSpMVAlg_t;

typedef enum {
  ALPHA_SPARSE_SPSV_ALG_DEFAULT = 0, /**< Default SpSV algorithm for the given format. */
  ALPHA_SPARSE_SPSV_CSR_ALG1 = 1,
  ALPHA_SPARSE_SPSV_CSR_ALG2 = 2,
  ALPHA_SPARSE_SPSV_CSR_ALG3 = 3,
  ALPHA_SPARSE_SPSV_CSR_ALG4 = 4,
  ALPHA_SPARSE_SPSV_CSR_ALG5 = 5,
  ALPHA_SPARSE_SPSV_CSR_ALG6 = 6,
  ALPHA_SPARSE_SPSV_CSR_ALG7 = 7,
  ALPHA_SPARSE_SPSV_CSR_ALG8 = 8,
} alphasparseSpSVAlg_t;

typedef enum {
  ALPHA_SPARSE_SPGEMM_STAGE_AUTO = 0,        /**< Automatic stage detection. */
  ALPHA_SPARSE_SPGEMM_STAGE_BUFFER_SIZE = 1, /**< Returns the required buffer size. */
  ALPHA_SPARSE_SPGEMM_STAGE_NNZ = 2,         /**< Computes number of non-zero entries. */
  ALPHA_SPARSE_SPGEMM_STAGE_COMPUTE = 3      /**< Performs the actual SpGEMM computation. */
} alphasparse_spgemm_stage_t;

typedef enum {
  ALPHA_SPARSE_SPGEMM_ALG_DEFAULT = 0 /**< Default SpGEMM algorithm for the given format. */
} alphasparse_spgemm_alg_t;

typedef enum {
  ALPHASPARSE_SPMM_ALG_DEFAULT = 0, /**< Default SpMM algorithm for the given format. */
  ALPHASPARSE_SPMM_COO_ALG1 = 1,
  ALPHASPARSE_SPMM_COO_ALG2 = 2,
  ALPHASPARSE_SPMM_COO_ALG3 = 3,
  ALPHASPARSE_SPMM_COO_ALG4 = 4,
  ALPHASPARSE_SPMM_CSR_ALG1 = 5,
  ALPHASPARSE_SPMM_CSR_ALG2 = 6,
  ALPHASPARSE_SPMM_CSR_ALG3 = 7,
  ALPHASPARSE_SPMM_CSR_ALG4 = 8,
  ALPHASPARSE_SPMM_CSR_ALG5 = 9,
  ALPHASPARSE_SPMM_CSR_ALG6 = 10,
} alphasparseSpMMAlg_t;

typedef enum {
  ALPHASPARSE_SPSM_ALG_DEFAULT = 0, /**< Default SpSM algorithm for the given format. */
  ALPHASPARSE_SPSM_COO_ALG1 = 1,
  ALPHASPARSE_SPSM_COO_ALG2 = 2,
  ALPHASPARSE_SPSM_COO_ALG3 = 3,
  ALPHASPARSE_SPSM_COO_ALG4 = 4,
  ALPHASPARSE_SPSM_CSR_ALG1 = 5,
  ALPHASPARSE_SPSM_CSR_ALG2 = 6,
  ALPHASPARSE_SPSM_CSR_ALG3 = 7,
  ALPHASPARSE_SPSM_CSR_ALG_ROC = 8,
  ALPHASPARSE_SPSM_CSR_ALG_CW = 9,
  ALPHASPARSE_SPSM_CSR_ALG_MY = 10,
} alphasparseSpSMAlg_t;

typedef enum {
  ALPHASPARSE_SDDMM_ALG_DEFAULT = 0, /**< Default SpSM algorithm for the given format. */
} alphasparseSDDMMAlg_t;

typedef enum {
  ALPHASPARSE_SPGEMM_DEFAULT = 0, /**< Default SpGEMM algorithm for the given format. */
  ALPHASPARSE_SPGEMM_ALG1 = 1,
  ALPHASPARSE_SPGEMM_ALG2 = 2,
  ALPHASPARSE_SPGEMM_ALG3 = 3,
} alphasparseSpGEMMAlg_t;

//old version
/* status of the routines */
// typedef enum {
//   ALPHA_SPARSE_STATUS_SUCCESS = 0,          /* the operation was successful */
//   ALPHA_SPARSE_STATUS_NOT_INITIALIZED = 1,  /* empty handle or matrix arrays */
//   ALPHA_SPARSE_STATUS_ALLOC_FAILED = 2,     /* internal error: memory allocation failed */
//   ALPHA_SPARSE_STATUS_INVALID_VALUE = 3,    /* invalid input value */
//   ALPHA_SPARSE_STATUS_EXECUTION_FAILED = 4, /* e.g. 0-diagonal element for triangular solver, etc. */
//   ALPHA_SPARSE_STATUS_INTERNAL_ERROR = 5,   /* internal error */
//   ALPHA_SPARSE_STATUS_NOT_SUPPORTED = 6,    /* e.g. operation for double precision doesn't support other types */
//   ALPHA_SPARSE_STATUS_INVALID_POINTER = 7,  /* e.g. invlaid pointers for DCU */
//   ALPHA_SPARSE_STATUS_INVALID_HANDLE = 8,   /* e.g. invlaid handle for DCU */
//   ALPHA_SPARSE_STATUS_INVALID_KERNEL_PARAM = 9 /* e.g. invalid param for sell-c-sigma gemv kernel */
// } alphasparseStatus_t;

/* sparse matrix operations */
// typedef enum {
//   ALPHA_SPARSE_OPERATION_NON_TRANSPOSE = 0,
//   ALPHA_SPARSE_OPERATION_TRANSPOSE = 1,
//   ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
// } alphasparseOperation_t;

// #define ALPHA_SPARSE_OPERATION_NUM 3

/* sparse matrix indexing: C-style or Fortran-style */
// typedef enum {
//   ALPHA_SPARSE_INDEX_BASE_ZERO = 0, /* C-style */
//   ALPHA_SPARSE_INDEX_BASE_ONE = 1   /* Fortran-style */
// } alphasparseIndexBase_t;

// #define ALPHA_SPARSE_INDEX_NUM 2
/*
 * ict sparse matirx implement;
 * ----------------------------------------------------------------------------------------------------------------------
 */
// #ifndef COMPLEX
// #ifndef DOUBLE
// #define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_FLOAT
// #else
// #define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_DOUBLE
// #endif
// #else
// #ifndef DOUBLE
// #define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX
// #else
// #define ALPHA_SPARSE_DATATYPE ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX
// #endif
// #endif
// typedef void *alpha_internal_vector;
// struct alphasparse_inspector__;
// #ifdef __cplusplus
// typedef alphasparse_inspector__ *alphasparse_inspector_t;
// #endif
// typedef struct alphasparse_inspector__ *alphasparse_inspector_t;

// typedef struct {
//   internal_spmat mat;
//   alphasparseFormat_t format;        // csr,coo,csc,bsr,ell,dia,sky...
//   alphasparse_datatype_t datatype;    // s,d,c,z
//   alphasparse_inspector_t inspector;  // for autotuning
//   void *dcu_info;                     // for dcu autotuning, alphasparse_dcu_mat_info_t
// } alphasparse_matrix;

// typedef alphasparse_matrix *alphasparse_matrix_t;

typedef struct {
  alpha_internal_vector vec;
  alphasparse_datatype_t datatype;    // s,d,c,z
  int64_t size;
} alphasparse_vector;

typedef alphasparse_vector *alphasparse_DnVec_Descr_t;

/*structures for DCU usages*/
struct alpha_dcu_matrix_descr {
  alphasparse_matrix_type_t
      type; /* matrix type: general, diagonal or triangular / symmetric / hermitian */
  alphasparse_fill_mode_t mode; /* upper or lower triangular part of the matrix ( for triangular /
                                  symmetric / hermitian case) */
  alphasparse_diag_type_t
      diag; /* unit or non-unit diagonal ( for triangular / symmetric / hermitian case) */
  alphasparseIndexBase_t base; /* C-style or Fortran-style*/
};

typedef struct alpha_dcu_matrix_descr *alpha_dcu_matrix_descr_t;
typedef enum {
  ALPHA_SPARSE_DCU_POINTER_MODE_HOST = 0,
  ALPHA_SPARSE_DCU_POINTER_MODE_DVICE = 1,
} alphasparse_dcu_pointer_mode_t;

typedef enum {
  ALPHA_SPARSE_DCU_LAYER_MODE_NONE = 0,      /**< layer is not active. */
  ALPHA_SPARSE_DCU_LAYER_MODE_LOG_TRACE = 1, /**< layer is in logging mode. */
  ALPHA_SPARSE_DCU_LAYER_MODE_LOG_BENCH = 2  /**< layer is in benchmarking mode. */
} alphasparse_dcu_layer_mode_t;

typedef enum {
  ALPHA_SPARSE_DCU_INDEXTYPE_U16 = 1, /**< 16 bit unsigned integer. */
  ALPHA_SPARSE_DCU_INDEXTYPE_I32 = 2, /**< 32 bit signed integer. */
  ALPHA_SPARSE_DCU_INDEXTYPE_I64 = 3  /**< 64 bit signed integer. */
} alphasparse_dcu_indextype_t;

typedef enum {
  ALPHA_SPARSE_DCU_HYB_PARTITION_AUTO = 0, /**< automatically decide on ELL nnz per row. */
  ALPHA_SPARSE_DCU_HYB_PARTITION_USER = 1, /**< user given ELL nnz per row. */
  ALPHA_SPARSE_DCU_HYB_PARTITION_MAX = 2   /**< max ELL nnz per row, no COO part. */
} alphasparse_dcu_hyb_partition_t;
typedef enum {
  ALPHA_SPARSE_DCU_ANALYSIS_POLICY_REUSE = 0, /**< try to re-use meta data. */
  ALPHA_SPARSE_DCU_ANALYSIS_POLICY_FORCE = 1  /**< force to re-build meta data. */
} alphasparse_dcu_analysis_policy_t;

typedef enum {
  ALPHA_SPARSE_DCU_SOLVE_POLICY_AUTO = 0 /**< automatically decide on level information. */
} alphasparse_dcu_solve_policy_t;

typedef enum {
  ALPHA_SPARSE_DCU_ACTION_SYMBOLIC = 0, /**< Operate only on indices. */
  ALPHA_SPARSE_DCU_ACTION_NUMERIC = 1   /**< Operate on data and indices. */
} alphasparse_dcu_action_t;

typedef enum {
  ALPHA_SPARSE_DCU_DENSE_TO_SPARSE_ALG_DEFAULT =
      0, /**< Default dense to sparse algorithm for the given format. */
} alphasparse_dcu_dense_to_sparse_alg_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPARSE_TO_DENSE_ALG_DEFAULT =
      0, /**< Default sparse to dense algorithm for the given format. */
} alphasparse_dcu_sparse_to_dense_alg_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPMV_ALG_DEFAULT = 0, /**< Default SpMV algorithm for the given format. */
  ALPHA_SPARSE_DCU_SPMV_ALG_COO = 1,     /**< COO SpMV algorithm for COO matrices. */
  ALPHA_SPARSE_DCU_SPMV_ALG_CSR_ADAPTIVE = 2,  /**< CSR SpMV algorithm 1 (adaptive) for CSR matrices. */
  ALPHA_SPARSE_DCU_SPMV_ALG_CSR_STREAM = 3, /**< CSR SpMV algorithm 2 (stream) for CSR matrices. */
  ALPHA_SPARSE_DCU_SPMV_ALG_ELL = 4         /**< ELL SpMV algorithm for ELL matrices. */
} alphasparse_dcu_spmv_alg_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_AUTO = 0,        /**< Automatic stage detection. */
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_BUFFER_SIZE = 1, /**< Returns the required buffer size. */
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_NNZ = 2,         /**< Computes number of non-zero entries. */
  ALPHA_SPARSE_DCU_SPGEMM_STAGE_COMPUTE = 3      /**< Performs the actual SpGEMM computation. */
} alphasparse_dcu_spgemm_stage_t;

typedef enum {
  ALPHA_SPARSE_DCU_SPGEMM_ALG_DEFAULT = 0 /**< Default SpGEMM algorithm for the given format. */
} alphasparse_dcu_spgemm_alg_t;