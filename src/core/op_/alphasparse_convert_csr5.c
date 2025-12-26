#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/check.h"
#include "alphasparse/util/malloc.h"

#include <stdio.h>

alphasparse_status_t convert_csr5_datatype_csr(const alpha_internal_spmat *source,
                                               alpha_internal_spmat **dest,
                                               alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_csr5_s_csr((spmat_csr_s_t *)source, (spmat_csr5_s_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_csr5_d_csr((spmat_csr_d_t *)source, (spmat_csr5_d_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    return convert_csr5_c_csr((spmat_csr_c_t *)source, (spmat_csr5_c_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    return convert_csr5_z_csr((spmat_csr_z_t *)source, (spmat_csr5_z_t **)dest);
  } else {
    printf("convert_csr5_datatype_csr\n");
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t convert_csr5_datatype_format(
    const alpha_internal_spmat *source, alpha_internal_spmat **dest,
    alphasparse_datatype_t datatype, alphasparse_format_t format) {
  if (format == ALPHA_SPARSE_FORMAT_CSR) {
    return convert_csr5_datatype_csr(source, dest, datatype);
  }
  else {
    printf("convert_csr5_datatype_format\n");
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_csr5(
    const alphasparse_matrix_t source, /* convert original matrix to csr5 representation */
    const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  alphasparse_matrix *dest_ = alpha_malloc(sizeof(alphasparse_matrix));
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_CSR5;
  dest_->datatype = source->datatype;
  alphasparse_status_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_csr5_datatype_format(
        (const alpha_internal_spmat *)source->mat, (alpha_internal_spmat **)&dest_->mat, source->datatype, source->format);
  } else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else {
    printf("alphasparse_convert_csr5\n");
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}