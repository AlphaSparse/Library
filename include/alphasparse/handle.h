#pragma once

#ifdef __CUDA__
#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif
#ifdef __HIP__
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#endif
#include <vector>

/**
 *  @brief header for the internal sparse matrix definitions
 */

#include "spdef.h"
#include <assert.h>
#include <stdio.h>

#ifdef __HIP__
alphasparseStatus_t
get_alphasparse_status_for_hip_status(hipError_t status);
#endif
typedef struct alphasparse_handle_s // 为了能在内部引用自己，给它一个名字
{
  // device id
  int device;
  // device wavefront size
  int wavefront_size;
  // asic revision
  int asic_rev;
  // stream ; default stream is system stream NULL
#ifdef __CUDA__
  cudaDeviceProp properties;
  cudaStream_t stream;
  cudaStream_t streams[6];
#endif
#ifdef __HIP__
  hipDeviceProp_t properties;
  hipStream_t stream;
  hipStream_t streams[6];
#endif
  // pointer mode ; default mode is host
  alphasparse_pointer_mode_t pointer_mode;
  // logging mode
  alphasparse_layer_mode_t layer_mode;
  // device buffer
  size_t buffer_size;
  void *buffer;
  // device one
  float *sone;
  double *done;

  // for check
  bool check_flag;

  int *process;
  void* pflat_analysis_data;

  alphasparse_handle_s() {
      device = 0;
      wavefront_size = 0;
      asic_rev = 0;
#ifdef __HIP__
      stream = nullptr;
      for(int i = 0; i < 6; ++i) streams[i] = nullptr;
#endif
#ifdef __CUDA__
      stream = nullptr;
      for(int i = 0; i < 6; ++i) streams[i] = nullptr;
#endif
      pointer_mode = ALPHA_SPARSE_POINTER_MODE_HOST;
      layer_mode = ALPHA_SPARSE_LAYER_MODE_NONE;
      buffer_size = 0;
      buffer = nullptr;
      sone = nullptr;
      done = nullptr;
      check_flag = false;
      process = nullptr;
      pflat_analysis_data = nullptr;
  }

} alphasparse_handle;

typedef alphasparse_handle *alphasparseHandle_t;

/********************************************************************************
 * alphasparse_hyb_mat is a structure holding the alphasparse HYB matrix.
 * It must be initialized using alphasparse_create_hyb_mat() and the returned
 * handle must be passed to all subsequent library function calls that involve
 * the HYB matrix.
 * It should be destroyed at the end using alphasparse_destroy_hyb_mat().
 *******************************************************************************/
typedef struct
{
  // num rows
  int m;
  // num cols
  int n;

  // partition type
  alphasparse_hyb_partition_t partition;

  // ELL matrix part
  int ell_nnz;
  int ell_width;
  int *ell_col_ind;
  void *ell_val;

  // COO matrix part
  int coo_nnz;
  int *coo_row_ind;
  int *coo_col_ind;
  void *coo_val;
} _alphasparse_hyb_mat;

struct _alphasparse_trm_info
{
  // maximum non-zero entries per row
  int max_nnz;

  // device array to hold row permutation
  int *row_map;
  // device array to hold pointer to diagonal entry
  int *trm_diag_ind;
  // device pointers to hold transposed data
  int *trmt_perm;
  int *trmt_row_ptr;
  int *trmt_col_ind;

  // some data to verify correct execution
  int m;
  int nnz;
  const struct alpha_matrix_descr *descr;
  const int *trm_row_ptr;
  const int *trm_col_ind;
};

/********************************************************************************
 * alphasparse_csrmv_info is a structure holding the alphasparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * alphasparse_create_csrmv_info() routine. It should be destroyed at the end
 * alphasparse_destroy_csrmv_info().
 *******************************************************************************/
enum csrgemv_algo
{
  ALPHA_CSRMV_AUTO,
  ALPHA_CSRMV_SCALAR,        // one thread process one row
  ALPHA_CSRMV_VECTOR,        // one wavefront process one row
  ALPHA_CSRMV_ROW_PARTITION, // Assign consecutive rows to a wavefront
  ALPHA_CSRMV_ADAPTIVE,      // csr-adaptive
  ALPHA_CSRMV_MERGE,         // merge-based
  ALPHA_CSRMV_XXX
};

struct _alphasparse_csrmv_info
{
  // algo tune
  bool algo_tuning;
  csrgemv_algo algo;
  int iter;

  // data struct for csr-adaptive method
  size_t size;                    // num row blocks
  unsigned long long *row_blocks; // row blocks
  bool csr_adaptive_has_tuned;
  int32_t stream_num;
  int32_t vector_num;
  int32_t vectorL_num;

  // data struct for csr-rowpartition method
  int *partition;
  bool csr_rowpartition_has_tuned;

  // data struct for csr-merge method
  void *coordinate;
  void *reduc_val;
  void *reduc_row;
  int num_merge_tiles;
  bool csr_merge_has_tuned;

  // data xxx
  int *r_csr_row_ptr;
  int *r_csr_col_ind;
  int *r_row_indx;
  void *r_csr_val;

  // some data to verify correct execution
  alphasparseOperation_t trans;
  int m;
  int n;
  int nnz;
  const struct alpha_matrix_descr *descr;
  const void *csr_row_ptr;
  const void *csr_col_ind;
};

/********************************************************************************
 * alphasparse_csrgemm_info is a structure holding the alphasparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the alphasparse_create_csrgemm_info() routine. It should be destroyed at the
 * end using alphasparse_destroy_csrgemm_info().
 *******************************************************************************/
struct _alphasparse_csrgemm_info
{
  // Perform alpha * A * B
  bool mul;
  // Perform beta * D
  bool add;
};

/*! typedefs to opaque info structs */
typedef struct _alphasparse_mat_info *alphasparse_mat_info_t;
typedef struct _alphasparse_trm_info *alphasparse_trm_info_t;
typedef struct _alphasparse_csrmv_info *alphasparse_csrmv_info_t;
typedef struct _alphasparse_csrgemm_info *alphasparse_csrgemm_info_t;
typedef _alphasparse_hyb_mat *alphasparse_hyb_mat_t;

/********************************************************************************
 * alphasparse_mat_info is a structure holding the matrix info data that is
 * gathered during the analysis routines. It must be initialized by calling
 * alphasparse_create_mat_info() and the returned info structure must be passed
 * to all subsequent function calls that require additional information. It
 * should be destroyed at the end using alphasparse_destroy_mat_info().
 *******************************************************************************/
struct _alphasparse_mat_info
{
  // info structs
  alphasparse_trm_info_t bsrsv_upper_info;
  alphasparse_trm_info_t bsrsv_lower_info;
  alphasparse_trm_info_t bsrsvt_upper_info;
  alphasparse_trm_info_t bsrsvt_lower_info;
  alphasparse_trm_info_t bsric0_info;
  alphasparse_trm_info_t bsrilu0_info;

  alphasparse_csrmv_info_t csrmv_info;
  alphasparse_trm_info_t csric0_info;
  alphasparse_trm_info_t csrilu0_info;
  alphasparse_trm_info_t csrsv_upper_info;
  alphasparse_trm_info_t csrsv_lower_info;
  alphasparse_trm_info_t csrsvt_upper_info;
  alphasparse_trm_info_t csrsvt_lower_info;
  alphasparse_trm_info_t csrsm_upper_info;
  alphasparse_trm_info_t csrsm_lower_info;
  alphasparse_trm_info_t csrsmt_upper_info;
  alphasparse_trm_info_t csrsmt_lower_info;
  alphasparse_csrgemm_info_t csrgemm_info;

  // zero pivot for csrsv, csrsm, csrilu0, csric0
  int *zero_pivot;

  // numeric boost for ilu0
  int boost_enable;
  int use_double_prec_tol;
  const void *boost_tol;
  const void *boost_val;
};

/********************************************************************************
 * alphasparse_csrmv_info is a structure holding the alphasparse csrmv info
 * data gathered during csrmv_analysis. It must be initialized using the
 * alphasparse_create_csrmv_info() routine. It should be destroyed at the end
 * using alphasparse_destroy_csrmv_info().
 *******************************************************************************/
alphasparseStatus_t
alphasparse_create_csrmv_info(alphasparse_csrmv_info_t *info);

/********************************************************************************
 * Destroy csrmv info.
 *******************************************************************************/
alphasparseStatus_t
alphasparse_destroy_csrmv_info(alphasparse_csrmv_info_t info);

/********************************************************************************
 * alphasparse_trm_info is a structure holding the alphasparse bsrsv, csrsv,
 * csrsm, csrilu0 and csric0 data gathered during csrsv_analysis,
 * csrilu0_analysis and csric0_analysis. It must be initialized using the
 * alphasparse_create_trm_info() routine. It should be destroyed at the end
 * using alphasparse_destroy_trm_info().
 *******************************************************************************/
alphasparseStatus_t
alphasparse_create_trm_info(alphasparse_trm_info_t *info);

/********************************************************************************
 * Destroy trm info.
 *******************************************************************************/
alphasparseStatus_t
alphasparse_destroy_trm_info(alphasparse_trm_info_t info);

/********************************************************************************
 * alphasparse_check_trm_shared checks if the given trm info structure
 * shares its meta data with another trm info structure.
 *******************************************************************************/
bool alphasparse_check_trm_shared(const alphasparse_mat_info_t info,
                                  alphasparse_trm_info_t trm);

/********************************************************************************
 * alphasparse_csrgemm_info is a structure holding the alphasparse csrgemm
 * info data gathered during csrgemm_buffer_size. It must be initialized using
 * the alphasparse_create_csrgemm_info() routine. It should be destroyed at the
 * end using alphasparse_destroy_csrgemm_info().
 *******************************************************************************/
alphasparseStatus_t
alphasparse_create_csrgemm_info(alphasparse_csrgemm_info_t *info);

/********************************************************************************
 * Destroy csrgemm info.
 *******************************************************************************/
alphasparseStatus_t
alphasparse_destroy_csrgemm_info(alphasparse_csrgemm_info_t info);

#ifdef __CUDA__
alphasparseStatus_t
get_alphasparse_status_for_cuda_status(cudaError_t status);
#endif
#ifdef __HIP__
alphasparseStatus_t
get_alphasparse_status_for_cuda_status(hipError_t status);
#endif
/********************************************************************************
 * ELL format indexing
 *******************************************************************************/
#define ELL_IND_ROW(i, el, m, width) (el) * (m) + (i)
#define ELL_IND_EL(i, el, m, width) (el) + (width) * (i)
#define ELL_IND(i, el, m, width) ELL_IND_ROW(i, el, m, width)

struct _alphasparse_mat_descr
{
  // matrix type
  alphasparse_matrix_type_t type = ALPHA_SPARSE_MATRIX_TYPE_GENERAL;
  // fill mode
  alphasparse_fill_mode_t fill_mode = ALPHA_SPARSE_FILL_MODE_LOWER;
  // diagonal type
  alphasparse_diag_type_t diag_type = ALPHA_SPARSE_DIAG_NON_UNIT;
  // index base
  alphasparseIndexBase_t base = ALPHA_SPARSE_INDEX_BASE_ZERO;
  // maximum nnz per row
  int64_t max_nnz_per_row = 0;
  bool spgemm_reuse_flag = false;

  void *row_map = {};
  void *rcsr_row_ptr = {};
  void *rcsr_col_idx = {};
  void *rcsr_val = {};
  int64_t warp_num_len = 0;
  void *d_warp_num = {};
  void *d_level_ptr = {};
  void *h_chain_ptr = {};
  int64_t h_level_size = 0;
  int64_t h_chain_size = 0;
  void *in_degree = {};
  void *csr_row_idx = {};

  // hip
  void *csr_rdiag = {};
  void *rcsr_rdiag = {};
};

typedef struct _alphasparse_mat_descr *alphasparseMatDescr_t;

typedef enum
{
  ALPHA_SPARSE_SOLVE_POLICY_NO_LEVEL = 0,
  ALPHA_SPARSE_SOLVE_POLICY_USE_LEVEL = 1
} alphasparseSolvePolicy_t;

enum info
{
  ALPHA_SPARSE_OPAQUE = 0,
  ALPHA_SPARSE_TRANSPARENT = 1
};

typedef info alpha_bsrsv2Info_t;
typedef info alpha_bsrsm2Info_t;
typedef info alpha_csric02Info_t;
typedef info alpha_csrilu02Info_t;
typedef info alpha_bsrilu02Info_t;
typedef info alpha_bsric02Info_t;
typedef info alphasparseColorInfo_t;

typedef struct _alphasparse_mat_descr *alphasparse_mat_descr_t;
typedef struct _alphasparse_mat_descr *alphasparseSpSVDescr_t;
typedef struct _alphasparse_mat_descr *alphasparseSpSMDescr_t;
typedef struct _alphasparse_mat_descr *alphasparseSpGEMMDescr_t;

typedef enum
{
  ALPHASPARSE_DIRECTION_ROW = 0,
  ALPHASPARSE_DIRECTION_COLUMN = 1
} alphasparseDirection_t;

typedef struct _alphasparse_mat_descr *alphasparse_mat_descr_t;
struct _alphasparseSpMatDescr
{
  bool init{};

  mutable bool analysed{};

  int64_t rows{};
  int64_t cols{};
  int64_t nnz{};
  int64_t nwarps{};

  int *row_data{}; // potential risk, overflow
  int *col_data{};
  int *ind_data{};
  void *val_data{};

  const int *const_row_data{};
  const int *const_col_data{};
  const int *const_ind_data{};
  const void *const_val_data{};

  alphasparseIndexType_t row_type{};
  alphasparseIndexType_t col_type{};
  alphasparseDataType data_type{};

  alphasparseIndexBase_t idx_base{};
  alphasparseFormat_t format{};

  alphasparse_mat_descr_t descr{};
  alphasparse_mat_info_t info{};

  alphasparseDirection_t block_dir{};
  int64_t block_dim{};
  int64_t ell_cols{};
  int64_t ell_width{};

  int64_t batch_count{};
  int64_t batch_stride{};
  int64_t offsets_batch_stride{};
  int64_t columns_values_batch_stride{};
};

struct _alphasparse_spvec_descr
{
  bool init{};

  int64_t size{};
  int64_t nnz{};

  void *idx_data{};
  void *val_data{};

  alphasparseIndexType_t idx_type{};
  alphasparseDataType data_type{};

  alphasparseIndexBase_t idx_base{};
};

struct _alphasparse_spmat_descr
{
  bool init{};
  bool analysed{};

  int64_t rows{};
  int64_t cols{};
  int64_t nnz{};

  void *row_data{};
  void *col_data{};
  void *ind_data{};
  void *val_data{};

  alphasparseIndexType_t row_type{};
  alphasparseIndexType_t col_type{};
  alphasparseDataType data_type{};

  alphasparseIndexBase_t idx_base{};
  alphasparseFormat_t format{};

  struct alpha_matrix_descr *descr{};
  alphasparse_mat_info_t info{};
};

struct _alphasparse_dnvec_descr
{
  bool init{};

  int64_t size{};
  void *values{};
  alphasparseDataType data_type{};
};

struct _alphasparse_dnmat_descr
{
  bool init{};

  int64_t rows{};
  int64_t cols{};
  int64_t ld{};

  void *values{};

  alphasparseDataType data_type{};
  alphasparseOrder_t order{};
};

typedef struct _alphasparseSpMatDescr *alphasparseSpMatDescr_t;

typedef struct _alphasparse_spvec_descr *alphasparseSpVecDescr_t;
typedef struct _alphasparse_spmat_descr *alphasparse_spmat_descr_t;
typedef struct _alphasparse_dnvec_descr *alphasparseDnVecDescr_t;
typedef struct _alphasparse_dnmat_descr *alphasparseDnMatDescr_t;
typedef struct _alphasparse_dnmat_descr *alphasparse_dnmat_descr_t;

alphasparseStatus_t
alphasparse_create_mat_descr(alpha_matrix_descr_t *descr);
alphasparseStatus_t
alphasparse_create_mat_info(alphasparse_mat_info_t *info);
alphasparseStatus_t
alphasparse_destroy_mat_descr(alpha_matrix_descr_t descr);
alphasparseStatus_t
alphasparse_destroy_mat_info(alphasparse_mat_info_t info);
alphasparseStatus_t
alphasparse_destroy_trm_info(alphasparse_trm_info_t info);
alphasparseStatus_t
alphasparseGetHandle(alphasparseHandle_t *handle);
alphasparseStatus_t
alphasparse_destory_handle(alphasparseHandle_t handle);
alphasparseStatus_t
initHandle(alphasparseHandle_t *handle);
alphasparseStatus_t
init_handle(alphasparseHandle_t *handle);
alphasparseStatus_t
alphasparse_get_handle(alphasparseHandle_t *handle);
// alphasparseStatus_t
// alphasparse_create_mat_descr(alpha_dcu_matrix_descr_t * descr);

double
get_time_us(void);

double
get_avg_time(std::vector<double> times);

double
get_avg_time_2(std::vector<double> times);

void alphasparse_init_s_csr_laplace2d(int *row_ptr,
                                      int *col_ind,
                                      float *val,
                                      int dim_x,
                                      int dim_y,
                                      int &M,
                                      int &N,
                                      int &nnz,
                                      alphasparseIndexBase_t base);

void alphasparse_init_d_csr_laplace2d(int *row_ptr,
                                      int *col_ind,
                                      double *val,
                                      int dim_x,
                                      int dim_y,
                                      int &M,
                                      int &N,
                                      int &nnz,
                                      alphasparseIndexBase_t base);

// void
// alphasparse_init_c_csr_laplace2d(int* row_ptr,
//                                  int* col_ind,
//                                  cuFloatComplex* val,
//                                  int dim_x,
//                                  int dim_y,
//                                  int& M,
//                                  int& N,
//                                  int& nnz,
//                                  alphasparseIndexBase_t base);

// void
// alphasparse_init_z_csr_laplace2d(int* row_ptr,
//                                  int* col_ind,
//                                  cuDoubleComplex* val,
//                                  int dim_x,
//                                  int dim_y,
//                                  int& M,
//                                  int& N,
//                                  int& nnz,
//                                  alphasparseIndexBase_t base);

alphasparseStatus_t
alphasparseCreateSpVec(alphasparseSpVecDescr_t *descr,
                       int64_t size,
                       int64_t nnz,
                       void *indices,
                       void *values,
                       alphasparseIndexType_t idx_type,
                       alphasparseIndexBase_t idx_base,
                       alphasparseDataType data_type);

alphasparseStatus_t
alphasparseCreateDnVec(alphasparseDnVecDescr_t *descr,
                       int64_t size,
                       void *values,
                       alphasparseDataType data_type);

alphasparseStatus_t
alphasparseCreateDnMat(alphasparseDnMatDescr_t *dnMatDescr,
                       int64_t rows,
                       int64_t cols,
                       int64_t ld,
                       void *values,
                       alphasparseDataType valueType,
                       alphasparseOrder_t order);

alphasparseStatus_t
alphasparseSpSV_createDescr(alphasparseSpSVDescr_t *descr);

alphasparseStatus_t
alphasparseSpSM_createDescr(alphasparseSpSMDescr_t *descr);

alphasparseStatus_t
alphasparseSpGEMM_createDescr(alphasparseSpGEMMDescr_t *descr);

typedef enum
{
  ALPHASPARSE_SPMAT_FILL_MODE,
  ALPHASPARSE_SPMAT_DIAG_TYPE
} alphasparseSpMatAttribute_t;

alphasparseStatus_t
alphasparseSpMatSetAttribute(alphasparseSpMatDescr_t spMatDescr,
                             alphasparseSpMatAttribute_t attribute,
                             void *data,
                             size_t dataSize);

alphasparseStatus_t alphasparseCreateMatDescr(alphasparseMatDescr_t *descr);
alphasparseStatus_t alphasparseSetMatIndexBase(alphasparseMatDescr_t descr, alphasparseIndexBase_t base);
alphasparseStatus_t alphasparseSetMatFillMode(alphasparseMatDescr_t descr, alphasparse_fill_mode_t fill);
alphasparseStatus_t alphasparseSetMatDiagType(alphasparseMatDescr_t descr, alphasparse_diag_type_t diag);
