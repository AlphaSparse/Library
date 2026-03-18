#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <memory.h>
#include <stdlib.h>

#include "alphasparse/inspector.h"
#include "alphasparse/spdef.h"

#include "create_csc.hpp"

#define C_IMPL(ONAME, TYPE)                                                                   \
  alphasparseStatus_t ONAME(                                                                  \
      alphasparse_matrix_t *A,                                                                \
      const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */         \
      const ALPHA_INT rows, const ALPHA_INT cols, ALPHA_INT *cols_start, ALPHA_INT *cols_end, \
      ALPHA_INT *row_indx, TYPE *values)                                                      \
  {                                                                                           \
        return create_csc(A, indexing, rows, cols, cols_start, cols_end, row_indx, values);   \
  }
  
C_IMPL(alphasparse_s_create_csc, float);
C_IMPL(alphasparse_d_create_csc, double);
C_IMPL(alphasparse_c_create_csc, ALPHA_Complex8);
C_IMPL(alphasparse_z_create_csc, ALPHA_Complex16);
#undef C_IMPL

