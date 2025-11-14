#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

template <typename J>
alphasparseStatus_t alphasparse_export_coo_template(const alphasparse_matrix_t source,
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **row_indx,
                          ALPHA_INT **col_indx,
                          J **values,
                          ALPHA_INT *nnz)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    // check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_COO, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    internal_spmat mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *row_indx = mat->row_data;
    *col_indx = mat->col_data;
    *values = ((J*)mat->val_data);
    *nnz = mat->nnz;
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define C_IMPL(ONAME, TYPE)                                                                             \
    alphasparseStatus_t ONAME(const alphasparse_matrix_t source,                                        \
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */    \
                          ALPHA_INT *rows,                                                              \
                          ALPHA_INT *cols,                                                              \   
                          ALPHA_INT **row_indx,                                                         \
                          ALPHA_INT **col_indx,                                                         \
                          TYPE **values,                                                                \
                          ALPHA_INT *nnz)                                                               \
    {                                                                                                   \
        return alphasparse_export_coo_template(source, indexing, rows, cols, row_indx, col_indx, values, nnz);\
    }

C_IMPL(alphasparse_s_export_coo, float);
C_IMPL(alphasparse_d_export_coo, double);
C_IMPL(alphasparse_c_export_coo, ALPHA_Complex8);
C_IMPL(alphasparse_z_export_coo, ALPHA_Complex16);
#undef C_IMPL