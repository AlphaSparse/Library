#include "alphasparse/kernel.h"
#include "alphasparse/compute.h"
#include "format/transpose_bsr.hpp"
#include "format/transpose_conj_bsr.hpp"
#include "format/destroy_bsr.hpp"

template <typename TYPE>
alphasparseStatus_t spmm_bsr_conj(const internal_spmat A, const internal_spmat B, internal_spmat *matC)
{
    internal_spmat conjugated_mat;
    transpose_conj_bsr<TYPE>(A, &conjugated_mat); //��matת��
    alphasparseStatus_t status = spmm_bsr<TYPE>(conjugated_mat, B, matC); //�ٵ��ó˷�
    destroy_bsr(conjugated_mat);
    return status;
}
