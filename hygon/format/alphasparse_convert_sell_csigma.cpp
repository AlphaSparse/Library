#include "alphasparse/format.h"
#include "alphasparse/inspector.h"
#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include <iostream>  
#include "alphasparse/util/internal_check.h"
#include "alphasparse/util/malloc.h"
#include "alphasparse/types.h"
#include "convert_sell_csigma_coo.hpp"
#include "convert_sell_csigma_csr.hpp"
alphasparseStatus_t convert_sell_csigma_datatype_coo(const internal_spmat source,
                                                    const ALPHA_INT C, 
                                                    const ALPHA_INT SIGMA,
                                                    internal_spmat *dest,
                                                    alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_sell_csigma_coo<ALPHA_INT, float, _internal_spmat>(source, C, SIGMA, dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_sell_csigma_coo<ALPHA_INT, double, _internal_spmat>(source, C, SIGMA, dest);
  // } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
  //   return convert_sell_csigma_coo<ALPHA_INT, ALPHA_Complex8, _internal_spmat>(source, dest);
  // } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
  //   return convert_sell_csigma_coo<ALPHA_INT, ALPHA_Complex16, _internal_spmat>(source, dest);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}
alphasparseStatus_t convert_sell_csigma_datatype_csr(const internal_spmat source,
                                                    const ALPHA_INT C, 
                                                    const ALPHA_INT SIGMA,
                                                    internal_spmat *dest,
                                                    alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_sell_csigma_csr<ALPHA_INT, float, _internal_spmat>(source, C, SIGMA, dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_sell_csigma_csr<ALPHA_INT, double, _internal_spmat>(source, C, SIGMA, dest);
  // } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
  //   return convert_sell_csigma_coo<ALPHA_INT, ALPHA_Complex8, _internal_spmat>(source, dest);
  // } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
  //   return convert_sell_csigma_coo<ALPHA_INT, ALPHA_Complex16, _internal_spmat>(source, dest);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparseStatus_t convert_sell_datatype_format(const internal_spmat source,
                                                const ALPHA_INT C, 
                                                const ALPHA_INT SIGMA,
                                                internal_spmat *dest,
                                                alphasparse_datatype_t datatype,
                                                alphasparseFormat_t format) {
  if (format == ALPHA_SPARSE_FORMAT_COO) {
    // std::cout<<"coo";
    return convert_sell_csigma_datatype_coo(source, C, SIGMA, dest, datatype);
  }
  else if (format == ALPHA_SPARSE_FORMAT_CSR) {
    // std::cout<<"csr";
    return convert_sell_csigma_datatype_csr(source, C, SIGMA, dest, datatype);
  }
  else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}
alphasparseStatus_t alphasparse_convert_sell_csigma(
                                                const alphasparse_matrix_t source, /* convert original matrix to SELL_C_Sgima representation */
                                                const bool SHORT_BINNING, const ALPHA_INT C, 
                                                const ALPHA_INT SIGMA,
                                                const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
                                                alphasparse_matrix_t *dest){
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  if (source->format != ALPHA_SPARSE_FORMAT_CSR) {
    *dest = NULL;
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  alphasparse_matrix *dest_ = (alphasparse_matrix *)alpha_malloc(sizeof(alphasparse_matrix));
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_SELL_C_SIGMA;
  dest_->datatype_cpu = source->datatype_cpu;
  dest_->inspector = NULL;
  dest_->inspector = NULL;
  dest_->inspector = (alphasparse_inspector_t)alpha_malloc(sizeof(alphasparse_inspector));
  alphasparse_inspector *kernel_inspector = (alphasparse_inspector *)dest_->inspector;
  kernel_inspector->mv_inspector = NULL;
  kernel_inspector->request_kernel = ALPHA_NONE;
  kernel_inspector->mm_inspector = NULL;
  kernel_inspector->mmd_inspector = NULL;
  kernel_inspector->sv_inspector = NULL;
  kernel_inspector->sm_inspector = NULL;
  kernel_inspector->memory_policy = ALPHA_SPARSE_MEMORY_AGGRESSIVE;
  alphasparseStatus_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_sell_datatype_format((const internal_spmat )source->mat,
                                        C, SIGMA,
                                        (internal_spmat *)&dest_->mat,
                                        source->datatype_cpu,
                                        source->format);
  } else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    alphasparse_matrix_t AA;
    check_error_return(alphasparse_transpose(source, &AA));
    status =
        convert_sell_datatype_format((const internal_spmat )AA->mat,
                                    C, SIGMA,
                                    (internal_spmat *)&dest_->mat,
                                    AA->datatype_cpu,
                                    AA->format);
    alphasparse_destroy(AA);
    return status;
  } else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparseStatus_t alphasparse_convert_sell_csigma_internal(
    const alphasparse_matrix_t source,       
    const bool SHORT_BINNING, const ALPHA_INT C, const ALPHA_INT SIGMA,
    const alphasparseOperation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  return alphasparse_convert_sell_csigma(source,SHORT_BINNING, C, SIGMA, operation, dest);
}