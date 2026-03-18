#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "update_value_bsr_plain.hpp"

template <typename TYPE>
alphasparseStatus_t alphasparse_update_values_tempalte_plain(alphasparse_matrix_t A, 
                                            const ALPHA_INT nvalues, 
                                            const ALPHA_INT *indx, 
                                            const ALPHA_INT *indy, 
                                            TYPE *values)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);     
    if(A->format == ALPHA_SPARSE_FORMAT_BSR)                            
    {                                                                   
        return update_values_bsr_plain(A->mat, nvalues, indx, indy, values);
    }                                                                   
    else                                                                
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;                       
}


#define C_IMPL(ONAME, TYPE)                                             \
alphasparseStatus_t ONAME (alphasparse_matrix_t A,                      \
                        const ALPHA_INT nvalues,                        \
                        const ALPHA_INT *indx,                          \
                        const ALPHA_INT *indy,                          \
                        TYPE *values)                                   \
{                                                                       \
     return alphasparse_update_values_tempalte_plain(A, nvalues, indx, indy, values);\
}

C_IMPL(alphasparse_s_update_values_plain, float);
C_IMPL(alphasparse_d_update_values_plain, double);
C_IMPL(alphasparse_c_update_values_plain, ALPHA_Complex8);
C_IMPL(alphasparse_z_update_values_plain, ALPHA_Complex16);
#undef C_IMPL
