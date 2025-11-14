#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

template <typename TYPE>
alphasparseStatus_t spmm_bsr_trans(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat transposed_mat;
    transpose_bsr<TYPE>(A, &transposed_mat); //��matת��
    alphasparseStatus_t status = spmm_bsr<TYPE>(transposed_mat, B, matC); //�ٵ��ó˷�
    destroy_bsr(transposed_mat);
    return status;
}
