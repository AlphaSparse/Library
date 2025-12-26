#include "alphasparse/format.h"
#
#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/check.h"
#include "alphasparse/util/malloc.h"

alphasparse_status_t convert_sky_datatype_coo(const alpha_internal_spmat *source,
                                             alpha_internal_spmat **dest,
                                             const alphasparse_fill_mode_t fill,
                                             alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_sky_s_coo((spmat_coo_s_t *)source, (spmat_sky_s_t **)dest, fill);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_sky_d_coo((spmat_coo_d_t *)source, (spmat_sky_d_t **)dest, fill);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    return convert_sky_c_coo((spmat_coo_c_t *)source, (spmat_sky_c_t **)dest, fill);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    return convert_sky_z_coo((spmat_coo_z_t *)source, (spmat_sky_z_t **)dest, fill);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

// alphasparse_status_t convert_sky_datatype_csr(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest,const alphasparse_fill_mode_t fill, alphasparse_datatype_t datatype)
//{
//    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//    {
//        return convert_sky_s_csr((spmat_csr_s_t *)source, (spmat_csr_s_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//    {
//        return convert_sky_d_csr((spmat_csr_d_t *)source, (spmat_csr_d_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//    {
//        return convert_sky_c_csr((spmat_csr_c_t *)source, (spmat_csr_c_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//    {
//        return convert_sky_z_csr((spmat_csr_z_t *)source, (spmat_csr_z_t **)dest, fill);
//    }
//    else
//    {
//        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//    }
//}

// alphasparse_status_t convert_sky_datatype_csc(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest,const alphasparse_fill_mode_t fill, alphasparse_datatype_t datatype)
//{
//    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//    {
//        return convert_sky_s_csc((spmat_csc_s_t *)source, (spmat_csr_s_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//    {
//        return convert_sky_d_csc((spmat_csc_d_t *)source, (spmat_csr_d_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//    {
//        return convert_sky_c_csc((spmat_csc_c_t *)source, (spmat_csr_c_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//    {
//        return convert_sky_z_csc((spmat_csc_z_t *)source, (spmat_csr_z_t **)dest, fill);
//    }
//    else
//    {
//        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//    }
//}

// Ict_sparse_status_t convert_sky_datatype_bsr(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest,const alphasparse_fill_mode_t fill, alphasparse_datatype_t datatype)
//{
//    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//    {
//        return convert_sky_s_bsr((spmat_bsr_s_t *)source, (spmat_csr_s_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//    {
//        return convert_sky_d_bsr((spmat_bsr_d_t *)source, (spmat_csr_d_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//    {
//        return convert_sky_c_bsr((spmat_bsr_c_t *)source, (spmat_csr_c_t **)dest, fill);
//    }
//    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//    {
//        return convert_sky_z_bsr((spmat_bsr_z_t *)source, (spmat_csr_z_t **)dest, fill);
//    }
//    else
//    {
//        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//    }
//}

alphasparse_status_t convert_sky_datatype_format(const alpha_internal_spmat *source,
                                                alpha_internal_spmat **dest,
                                                const alphasparse_fill_mode_t fill,
                                                alphasparse_datatype_t datatype,
                                                alphasparse_format_t format) {
  if (format == ALPHA_SPARSE_FORMAT_COO) {
    return convert_sky_datatype_coo(source, dest, fill, datatype);
  } else if (format == ALPHA_SPARSE_FORMAT_CSR) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    // return convert_sky_datatype_csr(source, dest, fill, datatype);
  } else if (format == ALPHA_SPARSE_FORMAT_CSC) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    // return convert_sky_datatype_csc(source, dest, fill, datatype);
  } else if (format == ALPHA_SPARSE_FORMAT_BSR) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    // return convert_sky_datatype_bsr(source, dest, fill, datatype);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_sky(const alphasparse_matrix_t source,
                                           const alphasparse_operation_t operation,
                                           const alphasparse_fill_mode_t fill,
                                           alphasparse_matrix_t *dest) {
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  alphasparse_matrix *dest_ = alpha_malloc(sizeof(alphasparse_matrix));
  if (source->format != ALPHA_SPARSE_FORMAT_COO) {
    *dest = NULL;
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_SKY;
  dest_->datatype = source->datatype;

  alphasparse_status_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_sky_datatype_format((const alpha_internal_spmat *)source->mat,
                                       (alpha_internal_spmat **)&dest_->mat, fill, source->datatype,
                                       source->format);
  } else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    alphasparse_matrix_t AA;
    check_error_return(alphasparse_transpose(source, &AA));
    status = convert_sky_datatype_format((const alpha_internal_spmat *)AA->mat,
                                         (alpha_internal_spmat **)&dest_->mat, fill, AA->datatype,
                                         AA->format);
    alphasparse_destroy(AA);
    return status;
  } else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_sky_internal(const alphasparse_matrix_t source,
                                                    const alphasparse_operation_t operation,
                                                    const alphasparse_fill_mode_t fill,
                                                    alphasparse_matrix_t *dest) {
  // TODO fill mode
  return alphasparse_convert_sky(source, operation, ALPHA_SPARSE_FILL_MODE_LOWER, dest);
}