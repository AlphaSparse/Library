#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "set/set_value_csr.hpp"
#include "set/set_value_coo.hpp"
#include "set/set_value_bsr.hpp"
#include "set/set_value_csc.hpp"

template <typename J>
alphasparseStatus_t alphasparse_set_value_template (alphasparse_matrix_t A, 
                        const ALPHA_INT row, 
                        const ALPHA_INT col,
                        const J value)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    if(A->format == ALPHA_SPARSE_FORMAT_CSR)
    {
        return set_value_csr(A->mat, row, col, value);
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_CSC)
    {
        return set_value_csc(A->mat, row, col, value);
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_BSR)
    {
        return set_value_bsr(A->mat, row, col, value);
    }
    else if(A->format == ALPHA_SPARSE_FORMAT_COO)
    {
        return set_value_coo(A->mat, row, col, value);
    }
    else
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
}

#define C_IMPL(ONAME, TYPE)                                             \
    alphasparseStatus_t ONAME (alphasparse_matrix_t A,  \
                                    const ALPHA_INT row,                \
                                    const ALPHA_INT col,                \
                                    const TYPE value)                   \
    {                                                                   \
        return alphasparse_set_value_template(A, row, col, value);      \
    }   
C_IMPL(alphasparse_s_set_value, float);
C_IMPL(alphasparse_d_set_value, double);
C_IMPL(alphasparse_c_set_value, ALPHA_Complex8);
C_IMPL(alphasparse_z_set_value, ALPHA_Complex16);
#undef C_IMPL