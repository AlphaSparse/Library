#pragma once

#include <map>
#include <string>
#include "alphasparse.h"

std::map<hipsparseSpSVAlg_t, std::string> cuda_spsv_alg_map {
    {HIPSPARSE_SPSV_ALG_DEFAULT, "HIPSPARSE_SPSV_ALG_DEFAULT"},
};
std::map<hipsparseSpMVAlg_t, std::string> cuda_spmv_alg_map {
    {HIPSPARSE_MV_ALG_DEFAULT, "HIPSPARSE_MV_ALG_DEFAULT"},
    // {HIPSPARSE_CSRMV_ALG1, "HIPSPARSE_CSRMV_ALG1"},
    // {HIPSPARSE_CSRMV_ALG2, "HIPSPARSE_CSRMV_ALG2"},
};
std::map<hipsparseSpSMAlg_t, std::string> cuda_spsm_alg_map {
};

std::map<alphasparseSpSVAlg_t, std::string> alpha_spsv_alg_map {
    {ALPHA_SPARSE_SPSV_ALG_DEFAULT, "ALPHA_SPARSE_SPSV_ALG_DEFAULT"},
    {ALPHA_SPARSE_SPSV_CSR_ALG1, "ALPHA_SPARSE_SPSV_CSR_ALG1"},
};
std::map<alphasparseSpMVAlg_t, std::string> alpha_spmv_alg_map {
    {ALPHA_SPARSE_SPMV_ALG_VECTOR, "ALPHA_SPARSE_SPMV_ALG_VECTOR"},
    {ALPHA_SPARSE_SPMV_ROW_PARTITION, "ALPHA_SPARSE_SPMV_ROW_PARTITION"},
    {ALPHA_SPARSE_SPMV_ALG_LOAD_BALANCE, "ALPHA_SPARSE_SPMV_ALG_LOAD_BALANCE"},
    {ALPHA_SPARSE_SPMV_ALG_FLAT, "ALPHA_SPARSE_SPMV_ALG_FLAT"},
    {ALPHA_SPARSE_SPMV_ALG_LINE_ENHANCE, "ALPHA_SPARSE_SPMV_ALG_LINE_ENHANCE"}
};
std::map<alphasparseSpSMAlg_t, std::string> alpha_spsm_alg_map {
    {ALPHASPARSE_SPSM_ALG_DEFAULT, "ALPHASPARSE_SPSM_ALG_DEFAULT"},
    {ALPHASPARSE_SPSM_CSR_ALG_MY, "ALPHASPARSE_SPSM_CSR_ALG_MY"},
};

std::map<hipsparseOperation_t, std::string> cuda_op_map {
    {HIPSPARSE_OPERATION_NON_TRANSPOSE, "HIPSPARSE_OPERATION_NON_TRANSPOSE"}, 
    {HIPSPARSE_OPERATION_TRANSPOSE, "HIPSPARSE_OPERATION_TRANSPOSE"},
    {HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE, "HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE"}
};
std::map<alphasparseOperation_t, std::string> alpha_op_map {
    {ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, "ALPHA_SPARSE_OPERATION_NON_TRANSPOSE"},
    {ALPHA_SPARSE_OPERATION_TRANSPOSE, "ALPHA_SPARSE_OPERATION_TRANSPOSE"},
    {ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE, "ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE"}
};
std::map<alphasparseOperation_t, hipsparseOperation_t> alpha2cuda_op_map {
    {ALPHA_SPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE}, 
    {ALPHA_SPARSE_OPERATION_TRANSPOSE, HIPSPARSE_OPERATION_TRANSPOSE},
    {ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE, HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
};
std::map<hipsparseFillMode_t, std::string> cuda_fill_map {
    {HIPSPARSE_FILL_MODE_UPPER, "HIPSPARSE_FILL_MODE_UPPER"},
    {HIPSPARSE_FILL_MODE_LOWER, "HIPSPARSE_FILL_MODE_LOWER"}
};
std::map<alphasparse_fill_mode_t, std::string> alpha_fill_map {
    {ALPHA_SPARSE_FILL_MODE_UPPER, "ALPHA_SPARSE_FILL_MODE_UPPER"},
    {ALPHA_SPARSE_FILL_MODE_LOWER, "ALPHA_SPARSE_FILL_MODE_LOWER"}
};
std::map<alphasparse_fill_mode_t, hipsparseFillMode_t> alpha2cuda_fill_map {
    {ALPHA_SPARSE_FILL_MODE_UPPER, HIPSPARSE_FILL_MODE_UPPER},
    {ALPHA_SPARSE_FILL_MODE_LOWER, HIPSPARSE_FILL_MODE_LOWER}
};
std::map<hipsparseDiagType_t, std::string> cuda_diag_map {
    {HIPSPARSE_DIAG_TYPE_NON_UNIT, "HIPSPARSE_DIAG_TYPE_NON_UNIT"},
    {HIPSPARSE_DIAG_TYPE_UNIT, "HIPSPARSE_DIAG_TYPE_UNIT"}
};
std::map<alphasparse_diag_type_t, std::string> alpha_diag_map {
    {ALPHA_SPARSE_DIAG_NON_UNIT, "ALPHA_SPARSE_DIAG_NON_UNIT"},
    {ALPHA_SPARSE_DIAG_UNIT, "ALPHA_SPARSE_DIAG_UNIT"}
};
std::map<alphasparse_diag_type_t, hipsparseDiagType_t> alpha2cuda_diag_map {
    {ALPHA_SPARSE_DIAG_NON_UNIT, HIPSPARSE_DIAG_TYPE_NON_UNIT},
    {ALPHA_SPARSE_DIAG_UNIT, HIPSPARSE_DIAG_TYPE_UNIT}
};
std::map<hipsparseOrder_t, std::string> cuda_order_map {
    {HIPSPARSE_ORDER_ROW, "HIPSPARSE_ORDER_ROW"},
    {HIPSPARSE_ORDER_COL, "HIPSPARSE_ORDER_COLUMN"},
};
std::map<alphasparseOrder_t, std::string> alpha_order_map {
    {ALPHASPARSE_ORDER_ROW, "ALPHASPARSE_ORDER_ROW"},
    {ALPHASPARSE_ORDER_COL, "ALPHASPARSE_ORDER_COL"},
};
std::map<alphasparseOrder_t, hipsparseOrder_t> alpha2cuda_order_map {
    {ALPHASPARSE_ORDER_ROW, HIPSPARSE_ORDER_ROW},
    {ALPHASPARSE_ORDER_COL, HIPSPARSE_ORDER_COL},
};
std::map<hipDataType, std::string> cuda_datatype_map {
    {HIP_R_32F, "HIP_R_32F"},
    {HIP_R_64F, "HIP_R_64F"},
    {HIP_C_32F, "HIP_C_32F"},
    {HIP_C_64F, "HIP_C_64F"}
};
std::map<alphasparseDataType, std::string> alpha_datatype_map {
    {ALPHA_R_32F, "ALPHA_R_32F"},
    {ALPHA_R_64F, "ALPHA_R_64F"},
    {ALPHA_C_32F, "ALPHA_C_32F"},
    {ALPHA_C_64F, "ALPHA_C_64F"}
};
std::map<alphasparseDataType, hipDataType> alpha2cuda_datatype_map {
    {ALPHA_R_32F, HIP_R_32F},
    {ALPHA_R_64F, HIP_R_64F},
    {ALPHA_C_32F, HIP_C_32F},
    {ALPHA_C_64F, HIP_C_64F}
};


template<typename U>
alphasparseDataType get_alpha_datatype() {
  return U{};
}

template<>
alphasparseDataType get_alpha_datatype<double>() {
  return ALPHA_R_64F;
}

template<>
alphasparseDataType get_alpha_datatype<float>() {
  return ALPHA_R_32F;
}

template<>
alphasparseDataType get_alpha_datatype<hipDoubleComplex>() {
  return ALPHA_C_64F;
}

template<>
alphasparseDataType get_alpha_datatype<hipFloatComplex>() {
  return ALPHA_C_32F;
}

alphasparseSpSVAlg_t get_alpha_spsv_alg(int alg_num) {
  alphasparseSpSVAlg_t alpha_alg;
  switch (alg_num) {
    case 0: {
      alpha_alg = ALPHA_SPARSE_SPSV_ALG_DEFAULT;
      break;
    }
    case 1: {
      alpha_alg = ALPHA_SPARSE_SPSV_CSR_ALG1;
      break;
    }
    case 2: {
      alpha_alg = ALPHA_SPARSE_SPSV_CSR_ALG2;
      break;
    }
	case 3: {
	  alpha_alg = ALPHA_SPARSE_SPSV_CSR_ALG3;
	  break;
	}
    default: {
      break;
    }
  }
  return alpha_alg;
}

alphasparseSpMVAlg_t get_alpha_spmv_alg(int alg_num) {
  alphasparseSpMVAlg_t alpha_alg;
  switch (alg_num) {
    case 0: {
        alpha_alg = ALPHA_SPARSE_SPMV_ALG_VECTOR;
        break;
    }
    case 1: {
        alpha_alg = ALPHA_SPARSE_SPMV_ROW_PARTITION;
        break;
    }
    case 2: {
        alpha_alg = ALPHA_SPARSE_SPMV_ALG_LOAD_BALANCE;
        break;
    }
    case 3: {
        alpha_alg = ALPHA_SPARSE_SPMV_ALG_FLAT;
        break;
    }
    case 4: {
        alpha_alg = ALPHA_SPARSE_SPMV_ALG_LINE_ENHANCE;
        break;
    }
    default: {
        break;
    }
  }
  return alpha_alg;
}

alphasparseSpSMAlg_t get_alpha_spsm_alg(int alg_num) {
  alphasparseSpSMAlg_t alpha_alg;
  switch (alg_num) {
    case 0: {
        alpha_alg = ALPHASPARSE_SPSM_ALG_DEFAULT;
        break;
    }
    case 1: {
        alpha_alg = ALPHASPARSE_SPSM_CSR_ALG_MY;
        break;
    }
    default: {
        break;
    }
  }
  return alpha_alg;
}

const char* get_filename(const char *file) {
  const char* lastSlash = strrchr(file, '/');
  if (lastSlash != NULL) {
    return lastSlash + 1;
  } else {
    return file;
  }
  return NULL;
}

/*
    对角线有元素的方阵的左下角每行非零元个数统计
*/
template<typename T>
void get_coo_row_nnz_cnt(
    const T *coo_row_idx, 
    const T *coo_col_idx, 
    const T m,
    const T nnz,
    T *row_nnz_cnt
) {
    // printf("m: %d\n", m);
    T ptr = 0;
    for (int row = 0; row < m; row++) {
        T cnt = 0;
        while (ptr < nnz && row == coo_row_idx[ptr] && coo_row_idx[ptr] >= coo_col_idx[ptr]) {
            printf("%d,%d\n", row, coo_col_idx[ptr]);
            cnt++;
            ptr++;
        }
        // printf("%d,%d\n", row, cnt);
        row_nnz_cnt[row] = cnt;
        while (ptr < nnz && row == coo_row_idx[ptr]) {
            ptr++;
        }
    }
    return;
}

/*
    对角线有元素的方阵的左下角每行非零元个数统计
*/
template<typename T>
void get_csr_row_nnz_cnt(
    const T *csr_row_ptr, 
    const T *csr_col_idx, 
    const T m,
    const T nnz,
    T *row_nnz_cnt
) {
    for (int row = 0; row < m; row++) {
      int cnt = 0;
      for (int ptr = csr_row_ptr[row]; ptr < csr_row_ptr[row + 1] && csr_col_idx[ptr] <= row; ptr++) {
        cnt++;
      }
    //   printf("row: %d, cnt: %d\n", row, cnt);
      row_nnz_cnt[row] = cnt;
    }
    return;
}

// [row_start, row_end)
template<typename T>
T 
get_elem_cnt_all(
    const T row_start,
    const T row_end,
    const T *row_nnz_cnt
) {
    T cnt = 0;
    for (T row = row_start; row < row_end; row++) {
        cnt += row_nnz_cnt[row];
    }
    return cnt;
}


template<typename T>
void warp_divide(
    const T *row_nnz_cnt, 
    const T m, 
    const T border,
    T *len, 
    T *warp_num
) {
    const T WARP_SIZE = 32;
    warp_num[0] = 0;
    T row_end;
    T elem_cnt_all = 0;
    T k = 1;
    double elem_cnt_avg = 0;
    for (T row_start = 0; row_start < m; row_start += WARP_SIZE) {
        row_end = row_start + WARP_SIZE;          // [row_start, row_end)
        row_end = (row_end > m) ? m : row_end;
        elem_cnt_all = get_elem_cnt_all(row_start, row_end, row_nnz_cnt);
        elem_cnt_avg = (double)elem_cnt_all / (row_end - row_start);
        if (elem_cnt_avg >= border) {  // warp-level
        for (T row_cur = row_start + 1; row_cur <= row_end; row_cur++) {
            warp_num[k] = row_cur;
            k++;
        }
        } else {
            warp_num[k] = row_end;
            k++;
        }
    }
    *len = k;
    return;
}




// csr format
template<typename U>
void print_hipsparse_matA(
    hipsparseSpMatDescr_t matA
) {
    int64_t m, n;
    int64_t nnz;
    int64_t *csrRowOffsets;
    int64_t *csrColInd;
    U *csrValues;
    hipsparseIndexType_t csrRowOffsetsType;
    hipsparseIndexType_t csrColIndType;
    hipsparseIndexBase_t idxBase;
    hipDataType valueType;
    hipsparseCsrGet(matA,
                &m,
                &n,
                &nnz,
                (void **)&csrRowOffsets,
                (void **)&csrColInd,
                (void **)&csrValues,
                &csrRowOffsetsType,
                &csrColIndType,
                &idxBase,
                &valueType);
    int32_t *csr_row_ptr = (int32_t *)malloc(sizeof(int32_t) * (m + 1));
    int32_t *csr_col_idx = (int32_t *)malloc(sizeof(int32_t) * nnz);
    U *csr_val = (U *)malloc(sizeof(U) * nnz);
    hipMemcpy(csr_row_ptr, csrRowOffsets, sizeof(int32_t) * (m + 1), hipMemcpyDeviceToHost);
    hipMemcpy(csr_col_idx, csrColInd, sizeof(int32_t) * nnz, hipMemcpyDeviceToHost);
    hipMemcpy(csr_val, csrValues, sizeof(U) * nnz, hipMemcpyDeviceToHost);
    std::cout << "m: " << m << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "csr_row_ptr" << std::endl;
    for (int i = 0; i < m + 1; i++) {
        std::cout << csr_row_ptr[i] << " ";
    }
    std::cout << "\ncsr_col_idx\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_col_idx[i] << " ";
    }
    std::cout << "\ncsr_val\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_val[i] << "; ";
    }
    std::cout << std::endl;
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
}

// csr format
template<typename U>
void print_alphasparse_matA(
    alphasparseSpMatDescr_t matA
) {
    int m = matA->rows;
    int nnz = matA->nnz;
    int *csr_row_ptr = (int *)malloc(sizeof(int) * (m + 1));
    int *csr_col_idx = (int *)malloc(sizeof(int) * nnz);
    U *csr_val = (U *)malloc(sizeof(U) * nnz);
    hipMemcpy(csr_row_ptr, matA->row_data, sizeof(int) * (m + 1), hipMemcpyDeviceToHost);
    hipMemcpy(csr_col_idx, matA->col_data, sizeof(int) * nnz, hipMemcpyDeviceToHost);
    hipMemcpy(csr_val, matA->val_data, sizeof(U) * nnz, hipMemcpyDeviceToHost);
    std::cout << "m: " << m << std::endl;
    std::cout << "nnz: " << nnz << std::endl;
    std::cout << "csr_row_ptr" << std::endl;
    for (int i = 0; i < m + 1; i++) {
        std::cout << csr_row_ptr[i] << " ";
    }
    std::cout << "\ncsr_col_idx\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_col_idx[i] << " ";
    }
    std::cout << "\ncsr_val\n";
    for (int i = 0; i < nnz; i++) {
        std::cout << csr_val[i] << "; ";
    }
    std::cout << std::endl;
    free(csr_row_ptr);
    free(csr_col_idx);
    free(csr_val);
}
