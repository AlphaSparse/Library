#pragma once

#include "alphasparse/types.h"
#include "alphasparse/spdef.h"

void alphasparse_nnz_counter_coo(const ALPHA_INT *row_indx, const ALPHA_INT *col_indx, const ALPHA_INT nnz, ALPHA_INT *lo_p, ALPHA_INT *diag_p, ALPHA_INT *hi_p);
ALPHA_INT64 alphasparse_operations_mm(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_INT lo, const ALPHA_INT diag, const ALPHA_INT hi, const alphasparseOperation_t operation, const struct alpha_matrix_descr descr, const ALPHA_INT columns, const alphasparse_datatype_t datatype);
ALPHA_INT64 alphasparse_operations_mv(const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_INT lo, const ALPHA_INT diag, const ALPHA_INT hi, const alphasparseOperation_t operation, const struct alpha_matrix_descr descr, const alphasparse_datatype_t datatype);
