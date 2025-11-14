/**
 * @brief implement for alphasparse_transpose intelface
 * @author Zhuoqiang Guo <gzq9425@qq.com>
 */

#include "alphasparse/spapi.h"
#include "alphasparse/util.h"
#include "alphasparse/format.h"
#include "transpose_csr.hpp"
#include "transpose_coo.hpp"
#include "transpose_csc.hpp"

alphasparseStatus_t transpose_datatype_coo(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_coo<float>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_coo<double>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_coo<ALPHA_Complex8>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_coo<ALPHA_Complex16>(source, dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t transpose_datatype_csr(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_csr<float>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_csr<double>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_csr<ALPHA_Complex8>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_csr<ALPHA_Complex16>(source, dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t transpose_datatype_csc(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_csc<float>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_csc<double>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex8>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex16>(source, dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t transpose_datatype_bsr(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_csc<float>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_csc<double>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex8>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex16>(source, dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t transpose_datatype_sky(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_csc<float>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_csc<double>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex8>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex16>(source, dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t transpose_datatype_dia(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype)
{
    if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT)
    {
        return transpose_csc<float>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE)
    {
        return transpose_csc<double>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_FLOAT_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex8>(source, dest);
    }
    else if (datatype == ALPHA_SPARSE_DATATYPE_DOUBLE_COMPLEX)
    {
        return transpose_csc<ALPHA_Complex16>(source, dest);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t transpose_datatype_format(const internal_spmat source, internal_spmat *dest, alphasparse_datatype_t datatype, alphasparseFormat_t format)
{
    if (format == ALPHA_SPARSE_FORMAT_CSR)
    {
        return transpose_datatype_csr(source, dest, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_COO)
    {
        return transpose_datatype_coo(source, dest, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_CSC)
    {
        return transpose_datatype_csc(source, dest, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_BSR)
    {
        return transpose_datatype_bsr(source, dest, datatype);
    }
    else if (format == ALPHA_SPARSE_FORMAT_SKY)
    {
        return ALPHA_SPARSE_STATUS_NOT_SUPPORTED;
    }
    else if (format == ALPHA_SPARSE_FORMAT_DIA)
    {
        return transpose_datatype_dia(source, dest, datatype);
    }
    else
    {
        return ALPHA_SPARSE_STATUS_INVALID_VALUE;
    }
}

alphasparseStatus_t alphasparse_transpose(const alphasparse_matrix_t source, alphasparse_matrix_t *dest)
{
    check_null_return(source, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    check_null_return(source->mat, ALPHA_SPARSE_STATUS_NOT_INITIALIZED);
    alphasparse_matrix *dest_ = (alphasparse_matrix_t)alpha_malloc(sizeof(alphasparse_matrix));
    *dest = dest_;
    dest_->format = source->format;
    dest_->datatype_cpu = source->datatype_cpu;
    return transpose_datatype_format((const internal_spmat)source->mat, (internal_spmat *)&dest_->mat, source->datatype_cpu, source->format);
}