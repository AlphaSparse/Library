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
#include "coo_order.hpp"
#include <type_traits>

template <typename I, typename J>
alphasparseStatus_t create_sell_csigma(
    alphasparse_matrix_t *A,
    const alphasparseIndexBase_t indexing, /* indexing: C-style or Fortran-style */
    const I rows, const I cols, I *rows_start, I *rows_end,
    I *col_indx, J *values) {
  alphasparse_matrix *AA = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
  *A = AA;
  internal_spmat mat = (internal_spmat)alpha_malloc(sizeof(struct _internal_spmat));
  AA->format = ALPHA_SPARSE_FORMAT_SELL_C_SIGMA;
  if(std::is_same_v<J, float>)
    AA->datatype_cpu = ALPHA_SPARSE_DATATYPE_FLOAT;
  else if(std::is_same_v<J, double>)
    AA->datatype_cpu = ALPHA_SPARSE_DATATYPE_DOUBLE;
  else
    return ALPHA_SPARSE_STATUS_INVALID_VALUE;
  AA->mat = mat;

  return ALPHA_SPARSE_STATUS_SUCCESS;
}