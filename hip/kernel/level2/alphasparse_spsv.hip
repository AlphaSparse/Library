#include "../../format/transpose_csr.h"
#include "../../format/coo2csr.h"
#include "alphasparse.h"

#include "./spsv_csr_csc_cw/spsv_csc_n_lo_cw.h"
#include "./spsv_csr_csc_cw/spsv_csc_n_up_cw.h"
#include "./spsv_csr_csc_cw/spsv_csr_n_lo_cw.h"
#include "./spsv_csr_csc_cw/spsv_csr_n_up_cw.h"

#include "./spsv_csr_csc_cw/spsv_csc_u_lo_cw.h"
#include "./spsv_csr_csc_cw/spsv_csc_u_up_cw.h"
#include "./spsv_csr_csc_cw/spsv_csr_u_lo_cw.h"
#include "./spsv_csr_csc_cw/spsv_csr_u_up_cw.h"

#include "./spsv_csr_n_lo_tri_cublk.h"

#include "./spsv_csr_n_lo_nnz_balance.h"

#include <iostream>
#include <functional>
#include <tuple>
#include <map>

#define _REORDER true //是否矩阵行重排
#define _TOPO_SCHD true //是否按拓扑排序序列调度
#define _WPR true //是否一wavefront处理一行
#define _DYNAMIC_SCHD true //是否动态调度

typedef struct
{
    alphasparseFormat_t format;
    alphasparseOperation_t operation;
    alphasparse_fill_mode_t fillmode;
    alphasparse_diag_type_t diagtype;
    alphasparseDataType datatype;
} spsv_test_input;

template <typename T, typename U>
alphasparseStatus_t
spsv_template(alphasparseHandle_t handle,
              alphasparseOperation_t opA,
              const void *alpha,
              alphasparseSpMatDescr_t matA,
              alphasparseDnVecDescr_t vecX,
              alphasparseDnVecDescr_t vecY,
              alphasparseSpSVAlg_t alg,
              alphasparseSpSVDescr_t spsvDescr,
              void *externalBuffer)
{
    alphasparse_fill_mode_t cur_fill_mode = matA->descr->fill_mode;
    if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE || opA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        cur_fill_mode = (matA->descr->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER) ? ALPHA_SPARSE_FILL_MODE_UPPER : ALPHA_SPARSE_FILL_MODE_LOWER;
    }
    bool is_conj = (opA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE);
    if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, n_trans, lower, unit
        opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
        cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
        spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csr_u_lo_cw\n");
            spsv_csr_u_lo_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->row_data,
                (T *)matA->col_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, n_trans, lower, n_unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            // printf("spsv_csr_n_lo_cw\n");
            spsv_csr_n_lo_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->row_data,
                (T *)matA->col_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer);
            break;
        }
        case ALPHA_SPARSE_SPSV_CSR_ALG2:
        {
			// printf("cublk solve\n");
            spsv_csr_n_lo_tri_cublk_solve<_REORDER, _TOPO_SCHD, _DYNAMIC_SCHD, _WPR, T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->row_data,
                (T *)matA->col_data,
                (U *)spsvDescr->csr_rdiag,
                (U *)spsvDescr->rcsr_val,
                (T *)spsvDescr->rcsr_row_ptr,
                (T *)spsvDescr->rcsr_col_idx,
                (U *)spsvDescr->rcsr_rdiag,
                (T *)spsvDescr->row_map,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer);
            break;
        }
		case ALPHA_SPARSE_SPSV_CSR_ALG3:
		{
			if (_REORDER)
			{
				spsv_csr_n_lo_nnz_balance_solve<_REORDER, T, U>(
					handle,
					(T)matA->rows,
					(T)matA->nnz,
					*((U *)alpha),
					(U *)spsvDescr->rcsr_val,
					(T *)spsvDescr->csr_row_idx,
					(T *)spsvDescr->rcsr_col_idx,
					(T *)spsvDescr->row_map,
					(T *)spsvDescr->in_degree,
					(U *)vecX->values,
					(U *)vecY->values,
					externalBuffer);
			}
			else
			{
				spsv_csr_n_lo_nnz_balance_solve<_REORDER, T, U>(
					handle,
					(T)matA->rows,
					(T)matA->nnz,
					*((U *)alpha),
					(U *)matA->val_data,
					(T *)spsvDescr->csr_row_idx,
					(T *)matA->col_data,
					(T *)spsvDescr->row_map,
					(T *)spsvDescr->in_degree,
					(U *)vecX->values,
					(U *)vecY->values,
					externalBuffer);
			}
			break;
		}
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, n_trans, upper, unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csr_u_up_cw\n");
            spsv_csr_u_up_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->row_data,
                (T *)matA->col_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, n_trans, upper, n_unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csr_n_up_cw\n");
            spsv_csr_n_up_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->row_data,
                (T *)matA->col_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, trans || c_trans, lower, unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csc_u_lo_cw\n");
            spsv_csc_u_lo_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->col_data,
                (T *)matA->row_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer,
                is_conj);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, trans || c_trans, lower, n_unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csc_n_lo_cw\n");
            spsv_csc_n_lo_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->col_data,
                (T *)matA->row_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer,
                is_conj);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, trans || c_trans, upper, unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csc_u_up_cw\n");
            spsv_csc_u_up_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->col_data,
                (T *)matA->row_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer,
                is_conj);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_CSR && // csr, trans || c_trans, upper, n_unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        case ALPHA_SPARSE_SPSV_CSR_ALG1:
        {
            printf("spsv_csc_n_up_cw\n");
            spsv_csc_n_up_cw<T, U>(
                handle,
                (T)matA->rows,
                (T)matA->nnz,
                *((U *)alpha),
                (U *)matA->val_data,
                (T *)matA->col_data,
                (T *)matA->row_data,
                (U *)vecX->values,
                (U *)vecY->values,
                externalBuffer,
                is_conj);
            break;
        }
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, n_trans, lower, unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, n_trans, lower, non_unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, n_trans, upper, unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, n_trans, upper, non_unit
             opA == ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, trans || c_trans, lower, unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, trans || c_trans, lower, non_unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, trans || c_trans, upper, unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    else if (matA->format == ALPHA_SPARSE_FORMAT_COO && // coo, trans || c_trans, upper, non_unit
             opA != ALPHA_SPARSE_OPERATION_NON_TRANSPOSE &&
             cur_fill_mode == ALPHA_SPARSE_FILL_MODE_UPPER &&
             spsvDescr->diag_type == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        switch (alg)
        {
        default:
        {
            printf("No algorithm chose.\n");
            exit(1);
            break;
        }
        }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpSV_bufferSize(alphasparseHandle_t handle,
                           alphasparseOperation_t opA,
                           const void *alpha,
                           alphasparseSpMatDescr_t matA,
                           alphasparseDnVecDescr_t vecX,
                           alphasparseDnVecDescr_t vecY,
                           alphasparseDataType computeType,
                           alphasparseSpSVAlg_t alg,
                           alphasparseSpSVDescr_t spsvDescr,
                           size_t *bufferSize)
{
    size_t valTypeSize = 4;
    switch (computeType)
    {
    case ALPHA_R_32F:
    {
        valTypeSize = 4;
        break;
    }
    case ALPHA_R_64F:
    case ALPHA_C_32F:
    {
        valTypeSize = 8;
        break;
    }
    case ALPHA_C_64F:
    {
        valTypeSize = 16;
        break;
    }
    default:
    {
        break;
    }
    }
    size_t idxTypeSize = 4;
    // 参与spsv计算的矩阵为方阵，必有 mat->rows == mat->cols
    // 不论是否转置，分配的数组长度都一样
    size_t vecLen = matA->rows;
    size_t nnzCnt = matA->nnz;
    switch (alg)
    {
    case ALPHA_SPARSE_SPSV_ALG_DEFAULT:
    {
        *bufferSize = vecLen * valTypeSize    // diag mem
                      + vecLen * idxTypeSize; // get_val mem [get_val元素的数据类型是int32_t]
        break;
    }
    case ALPHA_SPARSE_SPSV_CSR_ALG1:
    {
        switch (opA)
        {
        case ALPHA_SPARSE_OPERATION_NON_TRANSPOSE:
        {
            *bufferSize = vecLen * idxTypeSize // get_val mem
                          + 1 * idxTypeSize;   // id_extractor mem
            break;
        }
        case ALPHA_SPARSE_OPERATION_TRANSPOSE:
        case ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        {
            *bufferSize = vecLen * valTypeSize   // tmp_sum mem
                          + vecLen * idxTypeSize // get_val mem
                          + 1 * idxTypeSize;     // id_extractor mem
            break;
        }
        default:
        {
            break;
        }
        }
        break;
    }
    case ALPHA_SPARSE_SPSV_CSR_ALG2:
    {
        hipMalloc((void **)&spsvDescr->row_map, (matA->rows) * idxTypeSize);
        hipMalloc((void **)&spsvDescr->csr_rdiag, (matA->rows) * valTypeSize);
        hipMalloc((void **)&spsvDescr->rcsr_row_ptr, (matA->rows + 1) * idxTypeSize);
        hipMalloc((void **)&spsvDescr->rcsr_col_idx, (matA->nnz) * idxTypeSize);
        hipMalloc((void **)&spsvDescr->rcsr_val, (matA->nnz) * valTypeSize);
        hipMalloc((void **)&spsvDescr->rcsr_rdiag, (matA->rows) * valTypeSize);
        *bufferSize = vecLen * idxTypeSize // get_value
                    + 1 * idxTypeSize;
        break;
    }
	case ALPHA_SPARSE_SPSV_CSR_ALG3:
	{
		hipMalloc((void **)&spsvDescr->csr_row_idx, (matA->nnz) * sizeof(int64_t));
		hipMalloc((void **)&spsvDescr->in_degree, (matA->rows) * sizeof(int64_t));
        if (_REORDER)
		{
			hipMalloc((void **)&spsvDescr->row_map, (matA->rows) * idxTypeSize);
			hipMalloc((void **)&spsvDescr->rcsr_row_ptr, (matA->rows + 1) * idxTypeSize);
			hipMalloc((void **)&spsvDescr->rcsr_col_idx, (matA->nnz) * idxTypeSize);
			hipMalloc((void **)&spsvDescr->rcsr_val, (matA->nnz) * valTypeSize);
			hipMalloc((void **)&spsvDescr->rcsr_rdiag, (matA->rows) * valTypeSize);
		}
		*bufferSize = vecLen * valTypeSize    // tmp_sum
					+ vecLen * idxTypeSize  // get_value
					+ vecLen * idxTypeSize; // in_degree
		break;
	}
    default:
    {
        break;
    }
    }
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpSV_solve(alphasparseHandle_t handle,
                      alphasparseOperation_t opA,
                      const void *alpha,
                      alphasparseSpMatDescr_t matA,
                      alphasparseDnVecDescr_t vecX,
                      alphasparseDnVecDescr_t vecY,
                      alphasparseDataType computeType,
                      alphasparseSpSVAlg_t alg,
                      alphasparseSpSVDescr_t spsvDescr,
                      void *externalBuffer)
{
    // if (matA->format == ALPHA_SPARSE_FORMAT_COO && alg == ALPHA_SPARSE_SPSV_ALG_DEFAULT) {
    // printf("***\n");
    if (matA->format == ALPHA_SPARSE_FORMAT_COO)
	{
        int m = matA->rows;
        int n = matA->cols;
        int nnz = matA->nnz;
        int *dCsrRowPtr = NULL;
        hipMalloc((void **)&dCsrRowPtr, sizeof(int) * (m + 1));
		alphasparseXcoo2csr(matA->row_data, nnz, m, dCsrRowPtr);
		alphasparseSpMatDescr_t matA_csr;
        alphasparseCreateCsr(
            &matA_csr,
            m,
            n,
            nnz,
            dCsrRowPtr,
            matA->col_data,
            matA->val_data,
            matA->row_type,
            matA->col_type,
            matA->idx_base,
            matA->data_type);
        hipMemcpy(matA->row_data, matA_csr->row_data, sizeof(int) * (m + 1), hipMemcpyDeviceToDevice);
        matA->format = ALPHA_SPARSE_FORMAT_CSR;
        alg = ALPHA_SPARSE_SPSV_CSR_ALG1;
    }
    // single real ; i32
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_R_32F)
    {
        return spsv_template<int32_t, float>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
    // double real ; i64
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_R_64F)
    {
        return spsv_template<int32_t, double>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
    // single complex ; i32
    /*
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_C_32F) {
        return spsv_template<int32_t, hipFloatComplex>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
    // double complex ; i32
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_C_64F) {
        return spsv_template<int32_t, hipDoubleComplex>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }*/
}

template <typename T, typename U>
alphasparseStatus_t
spsv_anaylsis_template(alphasparseHandle_t handle,
                       alphasparseOperation_t opA,
                       const void *alpha,
                       alphasparseSpMatDescr_t matA,
                       alphasparseDnVecDescr_t vecX,
                       alphasparseDnVecDescr_t vecY,
                       alphasparseSpSVAlg_t alg,
                       alphasparseSpSVDescr_t spsvDescr,
                       void *externalBuffer)
{
    alphasparse_fill_mode_t cur_fill_mode = matA->descr->fill_mode;
    if (opA == ALPHA_SPARSE_OPERATION_TRANSPOSE || opA == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    {
        cur_fill_mode = (matA->descr->fill_mode == ALPHA_SPARSE_FILL_MODE_LOWER) ? ALPHA_SPARSE_FILL_MODE_UPPER : ALPHA_SPARSE_FILL_MODE_LOWER;
    }
	switch (alg)
	{
	    case ALPHA_SPARSE_SPSV_CSR_ALG2:
		{
			// printf("cublk analysis\n");
			spsv_csr_n_lo_tri_cublk_analysis<_REORDER, _TOPO_SCHD, T, U>(
				handle,
	            (T)matA->rows,
		        (T)matA->nnz,
			    *((U *)alpha),
				(U *)matA->val_data,
	            (T *)matA->row_data,
		        (T *)matA->col_data,
			    (U *)spsvDescr->csr_rdiag,
				(T *)spsvDescr->row_map,
	            (T *)spsvDescr->rcsr_row_ptr,
		        (T *)spsvDescr->rcsr_col_idx,
			    (U *)spsvDescr->rcsr_val,
	            (U *)spsvDescr->rcsr_rdiag,
			    (U *)vecX->values,
				(U *)vecY->values,
			    externalBuffer);
		    break;
	    }
		case ALPHA_SPARSE_SPSV_CSR_ALG3:
		{
			spsv_csr_n_lo_nnz_balance_analysis<_REORDER>(
					handle,
					(T)matA->rows,
					(T)matA->nnz,
					*((U *)alpha),
					(T *)matA->row_data,
					(T *)matA->col_data,
					(U *)matA->val_data,
					(T *)spsvDescr->csr_row_idx,
					(T *)spsvDescr->in_degree,
					(T *)spsvDescr->row_map,
					(T *)spsvDescr->rcsr_row_ptr,
					(T *)spsvDescr->rcsr_col_idx,
					(U *)spsvDescr->rcsr_val,
					(U *)spsvDescr->rcsr_rdiag,
					(U *)vecX->values,
					(U *)vecY->values,
					externalBuffer);
			break;
		}
	}
    return ALPHA_SPARSE_STATUS_SUCCESS;
}

alphasparseStatus_t
alphasparseSpSV_analysis(alphasparseHandle_t handle,
                         alphasparseOperation_t opA,
                         const void *alpha,
                         alphasparseSpMatDescr_t matA,
                         alphasparseDnVecDescr_t vecX,
                         alphasparseDnVecDescr_t vecY,
                         alphasparseDataType computeType,
                         alphasparseSpSVAlg_t alg,
                         alphasparseSpSVDescr_t spsvDescr,
                         void *externalBuffer)
{
    spsvDescr->diag_type = matA->descr->diag_type;
    spsvDescr->fill_mode = matA->descr->fill_mode;
    spsvDescr->type = matA->descr->type;
    spsvDescr->base = matA->descr->base;

    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_R_32F)
    {
        return spsv_anaylsis_template<int32_t, float>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
    // double real ; i64
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_R_64F)
    {
        return spsv_anaylsis_template<int32_t, double>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
    /*
	// single complex ; i32
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_C_32F)
    {
        return spsv_anaylsis_template<int32_t, hipFloatComplex>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
    // double complex ; i32
    if (matA->row_type == ALPHA_SPARSE_INDEXTYPE_I32 && matA->data_type == ALPHA_C_64F)
    {
        return spsv_anaylsis_template<int32_t, hipDoubleComplex>(handle, opA, alpha, matA, vecX, vecY, alg, spsvDescr, externalBuffer);
    }
	*/

    return ALPHA_SPARSE_STATUS_SUCCESS;
}
