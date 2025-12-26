#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"
#include "alphasparse/util/check.h"
#include "alphasparse/util/malloc.h"

alphasparse_status_t convert_gebsr_datatype_coo(const alpha_internal_spmat *source,
                                               alpha_internal_spmat **dest,
                                               const ALPHA_INT block_row_dim,
                                               const ALPHA_INT block_col_dim,
                                               const alphasparse_layout_t block_layout,
                                               alphasparse_datatype_t datatype) {
  if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT) {
    return convert_gebsr_s_coo((spmat_coo_s_t *)source, (spmat_gebsr_s_t **)dest, block_row_dim,
                               block_col_dim, block_layout);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE) {
    return convert_gebsr_d_coo((spmat_coo_d_t *)source, (spmat_gebsr_d_t **)dest, block_row_dim,
                               block_col_dim, block_layout);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX) {
    return convert_gebsr_c_coo((spmat_coo_c_t *)source, (spmat_gebsr_c_t **)dest, block_row_dim,
                               block_col_dim, block_layout);
  } else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX) {
    return convert_gebsr_z_coo((spmat_coo_z_t *)source, (spmat_gebsr_z_t **)dest, block_row_dim,
                               block_col_dim, block_layout);
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

// alphasparse_status_t convert_gebsr_datatype_csr(const alpha_internal_spmat *source,
//                                              alpha_internal_spmat **dest,
//                                              const ALPHA_INT block_row_dim,
// const ALPHA_INT block_col_dim,
//                                              const alphasparse_layout_t block_layout,
//                                              alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_gebsr_s_csr((spmat_csr_s_t *)source, (spmat_gebsr_s_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_gebsr_d_csr((spmat_csr_d_t *)source, (spmat_gebsr_d_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_gebsr_c_csr((spmat_csr_c_t *)source, (spmat_gebsr_c_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_gebsr_z_csr((spmat_csr_z_t *)source, (spmat_gebsr_z_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparse_status_t convert_gebsr_datatype_csc(const alpha_internal_spmat *source,
//                                              alpha_internal_spmat **dest,
//                                              const ALPHA_INT block_row_dim,
// const ALPHA_INT block_col_dim,
//                                              const alphasparse_layout_t block_layout,
//                                              alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_gebsr_s_csc((spmat_csc_s_t *)source, (spmat_gebsr_s_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_gebsr_d_csc((spmat_csc_d_t *)source, (spmat_gebsr_d_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_gebsr_c_csc((spmat_csc_c_t *)source, (spmat_gebsr_c_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_gebsr_z_csc((spmat_csc_z_t *)source, (spmat_gebsr_z_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

// alphasparse_status_t convert_gebsr_datatype_gebsr(const alpha_internal_spmat *source,
//                                              alpha_internal_spmat **dest,
//                                              const ALPHA_INT block_row_dim,
// const ALPHA_INT block_col_dim,
//                                              const alphasparse_layout_t block_layout,
//                                              alphasparse_datatype_t datatype)
// {
//     if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
//     {
//         return convert_gebsr_s_gebsr((spmat_gebsr_s_t *)source, (spmat_gebsr_s_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
//     {
//         return convert_gebsr_d_gebsr((spmat_gebsr_d_t *)source, (spmat_gebsr_d_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
//     {
//         return convert_gebsr_c_gebsr((spmat_gebsr_c_t *)source, (spmat_gebsr_c_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
//     {
//         return convert_gebsr_z_gebsr((spmat_gebsr_z_t *)source, (spmat_gebsr_z_t **)dest,
//         block_row_dim, block_col_dim, block_layout);
//     }
//     else
//     {
//         return ALPHA_SPARSE_STATUS_INVALID_VALUE;
//     }
// }

alphasparse_status_t convert_gebsr_datatype_format(
    const alpha_internal_spmat *source, alpha_internal_spmat **dest, const ALPHA_INT block_row_dim,
    const ALPHA_INT block_col_dim, const alphasparse_layout_t block_layout,
    alphasparse_datatype_t datatype, alphasparse_format_t format) {
  if (format == ALPHA_SPARSE_FORMAT_COO) {
    return convert_gebsr_datatype_coo(source, dest, block_row_dim, block_col_dim, block_layout,
                                      datatype);
  }
  // else if (format == ALPHA_SPARSE_FORMAT_CSR)
  // {
  //     return convert_gebsr_datatype_csr(source, dest, block_row_dim, block_col_dim, block_layout,
  //     datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_CSC)
  // {
  //     return convert_gebsr_datatype_csc(source, dest, block_row_dim, block_col_dim, block_layout,
  //     datatype);
  // }
  // else if (format == ALPHA_SPARSE_FORMAT_GEBSR)
  // {
  //     return convert_gebsr_datatype_gebsr(source, dest, block_row_dim, block_col_dim,
  //     block_layout, datatype);
  // }
  else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}

alphasparse_status_t alphasparse_convert_gebsr(
    const alphasparse_matrix_t source, /* convert original matrix to GEBSR representation */
    const ALPHA_INT block_row_dim, const ALPHA_INT block_col_dim,
    const alphasparse_layout_t block_layout, /* block storage: row-major or column-major */
    const alphasparse_operation_t operation, /* as is, transposed or conjugate transposed */
    alphasparse_matrix_t *dest) {
  check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
  check_return(block_row_dim <= 0, ALPHA_SPARSE_STATUS_INVALID_VALUE);
  check_return(block_col_dim <= 0, ALPHA_SPARSE_STATUS_INVALID_VALUE);
  alphasparse_matrix *dest_ = alpha_malloc(sizeof(alphasparse_matrix));
  *dest = dest_;
  dest_->dcu_info = NULL;
  dest_->format = ALPHA_SPARSE_FORMAT_GEBSR;
  dest_->datatype = source->datatype;
  alphasparse_status_t status;

  if (operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE) {
    return convert_gebsr_datatype_format(
        (const alpha_internal_spmat *)source->mat, (alpha_internal_spmat **)&dest_->mat, block_row_dim,
        block_col_dim, block_layout, source->datatype, source->format);
  } else if (operation == ALPHA_SPARSE_OPERATION_TRANSPOSE) {
    alphasparse_matrix_t AA;
    check_error_return(alphasparse_transpose(source, &AA));
    status = convert_gebsr_datatype_format((const alpha_internal_spmat *)AA->mat,
                                           (alpha_internal_spmat **)&dest_->mat, block_row_dim,
                                           block_col_dim, block_layout, AA->datatype, AA->format);
    alphasparse_destroy(AA);
    return status;
  } else if (operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
    return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
  } else {
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  }
}