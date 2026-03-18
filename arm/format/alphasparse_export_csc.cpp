#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

template <typename J>
alphasparseStatus_t alphasparse_export_csc_template(const alphasparse_matrix_t source,
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT **cols_start,
                          ALPHA_INT **cols_end,
                          ALPHA_INT **row_indx,
                          J **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    // check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_CSC, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    internal_spmat mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *cols_start = mat->col_data;
    *cols_end = mat->col_data+1;
    *row_indx = mat->row_data;
    *values = ((J*)mat->val_data);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define C_IMPL(ONAME, TYPE)                                                                             \
    alphasparseStatus_t ONAME(const alphasparse_matrix_t source,                                        \
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */    \
                          ALPHA_INT *rows,                                                              \
                          ALPHA_INT *cols,                                                              \
                          ALPHA_INT **cols_start,                                                       \
                          ALPHA_INT **cols_end,                                                         \
                          ALPHA_INT **row_indx,                                                         \
                          TYPE **values)                                                                \
    {                                                                                                   \
        return alphasparse_export_csc_template(source, indexing, rows, cols, cols_start, cols_end, row_indx, values);\
    }

C_IMPL(alphasparse_s_export_csc, float);
C_IMPL(alphasparse_d_export_csc, double);
C_IMPL(alphasparse_c_export_csc, ALPHA_Complex8);
C_IMPL(alphasparse_z_export_csc, ALPHA_Complex16);
#undef C_IMPL