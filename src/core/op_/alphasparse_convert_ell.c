#include "alphasparse/format.h"
#include "alphasparse/spapi.h"
#include "alphasparse/spdef.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/check.h"
#include "alphasparse/util/malloc.h"

alphasparse_status_t convert_ell_datatype_coo(const alpha_internal_spmat *source,
                                             alpha_internal_spmat **dest,
                                             alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_ell_s_coo((spmat_coo_s_t *)source, (spmat_ell_s_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_ell_d_coo((spmat_coo_d_t *)source, (spmat_ell_d_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    return convert_ell_c_coo((spmat_coo_c_t *)source, (spmat_ell_c_t **)dest);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    return convert_ell_z_coo((spmat_coo_z_t *)source, (spmat_ell_z_t **)dest);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

// alphasparse_status_t convert_ell_datatype_ell(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_ell_s_ell((spmat_ell_s_t *)source, (spmat_ell_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_ell_d_ell((spmat_ell_d_t *)source, (spmat_ell_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_ell_c_ell((spmat_ell_c_t *)source, (spmat_ell_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_ell_z_ell((spmat_ell_z_t *)source, (spmat_ell_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparse_status_t convert_ell_datatype_csc(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_ell_s_csc((spmat_csc_s_t *)source, (spmat_ell_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_ell_d_csc((spmat_csc_d_t *)source, (spmat_ell_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_ell_c_csc((spmat_csc_c_t *)source, (spmat_ell_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_ell_z_csc((spmat_csc_z_t *)source, (spmat_ell_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparse_status_t convert_ell_datatype_bsr(const alpha_internal_spmat *source, alpha_internal_spmat
// **dest, alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_ell_s_bsr((spmat_bsr_s_t *)source, (spmat_ell_s_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_ell_d_bsr((spmat_bsr_d_t *)source, (spmat_ell_d_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_ell_c_bsr((spmat_bsr_c_t *)source, (spmat_ell_c_t **)dest);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_ell_z_bsr((spmat_bsr_z_t *)source, (spmat_ell_z_t **)dest);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

alphasparse_status_t convert_ell_datatype_format(const alpha_internal_spmat *source,
                                                alpha_internal_spmat **dest,
                                                alphasparse_datatype_t datatype,
                                                alphasparse_format_t format) {
  if (format == ALPHA_SPARSE_FORMAT_COO) {
    return convert_ell_datatype_coo(source, dest, datatype);
  }
  // else if (format == ALPHA_SPARSE_FORMAT_ELL)
  // {
  //     return convert_ell_datatype_ell(source, dest, datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_CSC)
  // {
  //     return convert_ell_datatype_csc(source, dest, datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_BSR)
  // {
  //     return convert_ell_datatype_bsr(source, dest, datatype);
  // }
  else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_ell(
    const alphasparse_matrix_t source,       /* convert original matrix to ELL representation */
    const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  if (source->format != ALPHA_SPARSE_FORMAT_COO) {
    *dest = NULL;
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  }
  alphasparse_matrix *dest_ = alpha_malloc(sizeof(alphasparse_matrix));
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_ELL;
  dest_->datatype = source->datatype;
  alphasparse_status_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_ell_datatype_format((const alpha_internal_spmat *)source->mat,
                                       (alpha_internal_spmat **)&dest_->mat, source->datatype,
                                       source->format);
  }
  // else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
  // {
  //     alphasparse_matrix_t AA;
  //     check_error_return(alphasparse_transpose(source, &AA));
  //     status = convert_ell_datatype_format((const alpha_internal_spmat *)AA->mat,
  //     (alpha_internal_spmat **)&dest_->mat, AA->datatype, AA->format); alphasparse_destroy(AA);
  //     return status;
  // }
  // else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
  // {
  //     return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  // }
  else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}