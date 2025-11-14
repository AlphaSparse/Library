#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

template <typename J>
alphasparseStatus_t alphasparse_export_ell_template(const alphasparse_matrix_t source,
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT *width,
                          ALPHA_INT **col_indx,
                          J **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    // check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_ELL, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    internal_spmat mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *rows = mat->rows;
    *cols = mat->cols;
    *width = mat->ell_width;
    *col_indx = (ALPHA_INT*)mat->ell_cols;
    *values = ((J*)mat->val_data);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define C_IMPL(ONAME, TYPE)                                                                         \
    alphasparseStatus_t ONAME(const alphasparse_matrix_t source,                                    \
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */\
                          ALPHA_INT *rows,                                                          \
                          ALPHA_INT *cols,                                                          \
                          ALPHA_INT *width,                                                         \
                          ALPHA_INT **col_indx,                                                     \
                          TYPE **values)                                                            \
    {                                                                                               \
        return alphasparse_export_ell_template(source, indexing, rows, cols, width, col_indx, values);\
    }

C_IMPL(alphasparse_s_export_ell, float);
C_IMPL(alphasparse_d_export_ell, double);
C_IMPL(alphasparse_c_export_ell, ALPHA_Complex8);
C_IMPL(alphasparse_z_export_ell, ALPHA_Complex16);
#undef C_IMPL