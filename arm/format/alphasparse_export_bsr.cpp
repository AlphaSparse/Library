#include "alphasparse.h"
#include "alphasparse/format.h"
#include "alphasparse/spmat.h"

template <typename J>
alphasparseStatus_t alphasparse_export_bsr_template(const alphasparse_matrix_t source,
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */
                          alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */
                          ALPHA_INT *rows,
                          ALPHA_INT *cols,
                          ALPHA_INT *block_size,
                          ALPHA_INT **rows_start,
                          ALPHA_INT **rows_end,
                          ALPHA_INT **col_indx,
                          J **values)
{
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);
    // check_return(source->datatype != ALPHA_SPARSE_DATATYPE, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(source->format != ALPHA_SPARSE_FORMAT_BSR, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    internal_spmat mat = source->mat;
    *indexing = ALPHA_SPARSE_INDEX_BASE_ZERO;
    *block_layout = (alphasparse_layout_t)mat->block_layout;
    *rows = mat->rows;
    *cols = mat->cols;
    *block_size = mat->block_dim;
    *rows_start = mat->row_data;
    *rows_end = mat->row_data+1;
    *col_indx = mat->col_data;
    *values = ((J*)mat->val_data);
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

#define C_IMPL(ONAME, TYPE)                                                                                         \
    alphasparseStatus_t ONAME(const alphasparse_matrix_t source,                                                    \
                          alphasparseIndexBase_t *indexing, /* indexing: C-style or Fortran-style */                \
                          alphasparse_layout_t *block_layout, /* block storage: row-major or column-major */        \
                          ALPHA_INT *rows,                                                                          \
                          ALPHA_INT *cols,                                                                          \
                          ALPHA_INT *block_size,                                                                    \
                          ALPHA_INT **rows_start,                                                                   \
                          ALPHA_INT **rows_end,                                                                     \
                          ALPHA_INT **col_indx,                                                                     \
                          TYPE **values)                                                                            \
    {                                                                                                               \
        return alphasparse_export_bsr_template(source, indexing, block_layout, rows, block_size, cols, rows_start, rows_end, col_indx, values);\
    }

C_IMPL(alphasparse_s_export_bsr, float);
C_IMPL(alphasparse_d_export_bsr, double);
C_IMPL(alphasparse_c_export_bsr, ALPHA_Complex8);
C_IMPL(alphasparse_z_export_bsr, ALPHA_Complex16);
#undef C_IMPL