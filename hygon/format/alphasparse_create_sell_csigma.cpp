#include <alphasparse/format.h>
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdlib.h>

#include "alphasparse/inspector.h"
#include "alphasparse/spdef.h"

#include "create_sell.hpp"

#define C_IMPL(ONAME, TYPE)                                                                   \
  alphasparseStatus_t ONAME(                                                                  \
      alphasparse_matrix_t *A,                                                                \
      const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */         \
      const ALPHA_INT rows, const ALPHA_INT cols, ALPHA_INT *rows_start, ALPHA_INT *rows_end, \
      ALPHA_INT *col_indx, TYPE *values)                                                      \
  {                                                                                           \
        return create_sell_csigma(A, indexing, rows, cols, rows_start, rows_end, col_indx, values);   \
  }
  
C_IMPL(alphasparse_s_create_sell_csigma, float);
C_IMPL(alphasparse_d_create_sell_csigma, double);
#undef C_IMPL