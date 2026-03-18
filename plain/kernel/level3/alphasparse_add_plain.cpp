#include "alphasparse/util.h"
#include "alphasparse/opt.h"
#include "alphasparse/spapi.h"
#include "alphasparse/kernel.h"
#include "alphasparse/spmat.h"
#include "add/add_csr_plain.hpp"
#include "add/add_csr_trans_plain.hpp"
#include "add/add_csr_conj_plain.hpp"

template <typename I = ALPHA_INT, typename J>
alphasparseStatus_t alphasparse_add_template_plain(const alphasparseOperation_t operation,
                                     const alphasparse_matrix_t A,
                                     const J alpha,
                                     const alphasparse_matrix_t B,
                                     alphasparse_matrix_t *matC)
{
    check_null_return(A->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(B->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(matC, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);

    check_return(A->format != B->format, ALPHA_SPARSE_STATUS_INVALID_VALUE);
    check_return(A->format != ALPHA_SPARSE_FORMAT_CSR, ALPHA_SPARSE_STATUS_NOT_SUPPORTED);

    alphasparse_matrix *AA = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
    *matC = AA;
    AA->format = A->format;
    AA->datatype_cpu = A->datatype_cpu;

    if(operation == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE)
        return add_csr_plain<J>(A->mat, alpha, B->mat, &(AA->mat));
    else if(operation == ALPHA_SPARSE_OPERATION_TRANSPOSE)
        return add_csr_trans_plain<J>(A->mat, alpha, B->mat, &(AA->mat));
    else if(operation == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
        return add_csr_conj_plain<J>(A->mat, alpha, B->mat, &(AA->mat));
    else
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
}

#define C_IMPL(ONAME, TYPE)                                             \
    alphasparseStatus_t ONAME(const alphasparseOperation_t operation,   \
                                        const alphasparse_matrix_t A,   \
                                        const TYPE alpha,               \
                                        const alphasparse_matrix_t B,   \
                                        alphasparse_matrix_t *matC)     \
    {                                                                   \
        return alphasparse_add_template_plain(operation, A, alpha, B, matC);  \
    }                                                   
C_IMPL(alphasparse_s_add_plain, float);
C_IMPL(alphasparse_d_add_plain, double);
C_IMPL(alphasparse_c_add_plain, ALPHA_Complex8);
C_IMPL(alphasparse_z_add_plain, ALPHA_Complex16);
#undef C_IMPL