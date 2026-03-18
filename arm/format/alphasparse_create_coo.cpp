// #include "alphasparse.h"
#include <alphasparse/opt.h>
#include <alphasparse/util.h>
#include <alphasparse/format.h>
#include <alphasparse/spapi.h>

#include <memory.h>
#include <stdlib.h>

#include "alphasparse/inspector.h"
#include "alphasparse/spdef.h"
#include "alphasparse/types.h"
#include "alphasparse/util/malloc.h"
#include "create_coo.hpp"

template <typename I, typename J>
alphasparseStatus_t alphasparse_create_coo_template(
    alphasparse_matrix_t *A,
    const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
    const I rows, const I cols, const I nnz, I *row_indx, I *col_indx,
    J *values) 
{  
  return create_coo<I, J>(A,            
                    indexing,     
                    rows,         
                    cols,         
                    nnz,          
                    row_indx,     
                    col_indx,     
                    values);      
}

#define C_IMPL(ONAME, TYPE)                                                         \
    alphasparseStatus_t ONAME(                                                      \
        alphasparse_matrix_t *A,                                                     \
        const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */ \
        const ALPHA_INT rows, const ALPHA_INT cols, const ALPHA_INT nnz,             \
        ALPHA_INT *row_indx, ALPHA_INT *col_indx,                                    \
        TYPE *values)                                                                \
    {                                                                                \
        return alphasparse_create_coo_template(A,                                    \
                                       indexing,                                     \
                                       rows,                                         \
                                       cols,                                         \
                                       nnz,                                          \
                                       row_indx,                                     \
                                       col_indx,                                     \
                                       values);                                      \
    }

C_IMPL(alphasparse_s_create_coo, float);
C_IMPL(alphasparse_d_create_coo, double);
C_IMPL(alphasparse_c_create_coo, ALPHA_Complex8);
C_IMPL(alphasparse_z_create_coo, ALPHA_Complex16);
#undef C_IMPL