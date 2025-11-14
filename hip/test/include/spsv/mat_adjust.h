#pragma once


/*
    description:
        coo格式的矩阵A作为输入，在A的基础上进行修改，将A裁剪成正方形，若对角线无元素则在该对角线位置补1
*/
template<typename T>
void mat_patch_trim_s(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    float **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    float *tmp_val = (float *)malloc(sizeof(float) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.f;
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.f;
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    float *new_val = (float *)malloc(sizeof(float) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(float) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}


template<typename T>
void mat_patch_trim_d(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    double **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    double *tmp_val = (double *)malloc(sizeof(double) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.0;
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = 1.0;
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    double *new_val = (double *)malloc(sizeof(double) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(double) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}

template<typename T>
void mat_patch_trim_c(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    hipFloatComplex **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    hipFloatComplex *tmp_val = (hipFloatComplex *)malloc(sizeof(hipFloatComplex) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_hipFloatComplex(1.f, 1.f);
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_hipFloatComplex(1.f, 1.f);
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    hipFloatComplex *new_val = (hipFloatComplex *)malloc(sizeof(hipFloatComplex) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(hipFloatComplex) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}

template<typename T>
void mat_patch_trim_z(
    T *m, 
    T *n, 
    T *nnz, 
    T **coo_row_idx, 
    T **coo_col_idx, 
    hipDoubleComplex **coo_val
) {
    int len = (*m > *n) ? *n : *m;      // 尽可能大地获取矩阵A的行列以取正方形矩阵
    T *tmp_row_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    T *tmp_col_idx = (T *)malloc(sizeof(T) * (*nnz + len));
    hipDoubleComplex *tmp_val = (hipDoubleComplex *)malloc(sizeof(hipDoubleComplex) * (*nnz + len));
    T tmp_nnz = 0;
    T ptr = 0;
    T cnt = 0;
    for (int row = 0; row < len; row++) {
        cnt = 0;    // 记录当前行有效NNZ数目
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row && (*coo_col_idx)[ptr] < row) {   
            // 对角线左侧位置NNZ进行存储
            tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
            tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
            tmp_val[tmp_nnz] = (*coo_val)[ptr];
            tmp_nnz++;
            cnt++;
            ptr++;
        }
        if (ptr >= (*nnz) || ((*coo_row_idx)[ptr] != row || (*coo_col_idx)[ptr] != row)) {   
            // 移动ptr过程中，扫描到当前行的对角线位置，发现此位置没有NNZ，需补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_hipDoubleComplex(1.f, 1.f);
            tmp_nnz++;
            cnt++;
        }
        while (ptr < (*nnz) && (*coo_row_idx)[ptr] == row) {    
            // 对角线右侧位置NNZ进行存储
            if ((*coo_col_idx)[ptr] < len) {
                tmp_row_idx[tmp_nnz] = (*coo_row_idx)[ptr];
                tmp_col_idx[tmp_nnz] = (*coo_col_idx)[ptr];
                tmp_val[tmp_nnz] = (*coo_val)[ptr];
                tmp_nnz++;
                cnt++;
            }
            ptr++;
        }
        if (cnt == 0) { 
            // 当前行没有NNZ，必须在对角线位置补充元素进行存储
            tmp_row_idx[tmp_nnz] = row;
            tmp_col_idx[tmp_nnz] = row;
            tmp_val[tmp_nnz] = make_hipDoubleComplex(1.f, 1.f);
            tmp_nnz++;
        }
    }
    T *new_row_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    T *new_col_idx = (T *)malloc(sizeof(T) * tmp_nnz);
    hipDoubleComplex *new_val = (hipDoubleComplex *)malloc(sizeof(hipDoubleComplex) * tmp_nnz);
    memcpy(new_row_idx, tmp_row_idx, sizeof(T) * tmp_nnz);
    memcpy(new_col_idx, tmp_col_idx, sizeof(T) * tmp_nnz);
    memcpy(new_val, tmp_val, sizeof(hipDoubleComplex) * tmp_nnz);
    free(tmp_row_idx);
    free(tmp_col_idx);
    free(tmp_val);
    free(*coo_row_idx);
    free(*coo_col_idx);
    free(*coo_val);
    *coo_row_idx = new_row_idx;
    *coo_col_idx = new_col_idx;
    *coo_val = new_val;
    *m = len;
    *n = len;
    *nnz = tmp_nnz;
    return;
}

/*
    description:
        coo格式的矩阵A每行元素求和归一化
        a_{ii} = \sum_{j = 0}^{n - 1} {a_{ij}} + 1, for all i in [0, m)
        a_{ij} /= a_{ii}, for all {i, j} in [{0, 0}, {m, n})
*/

void mat_adjust_nnz_s(
    const int *coo_row_index,
    const int *coo_col_index,
    float *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum = 0.f;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                    diag_ptr = j;
                }
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            coo_values[diag_ptr] = partial_sum + 1;
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum = 0.f;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    }
}


void mat_adjust_nnz_d(
    const int *coo_row_index,
    const int *coo_col_index,
    double *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum = 0.0;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                  diag_ptr = j;
                }
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            coo_values[diag_ptr] = partial_sum + 1;
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum = 0.f;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum += fabs(coo_values[j]);
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] /= partial_sum;
            }
        }
    }
}


void mat_adjust_nnz_c(
    const int *coo_row_index,
    const int *coo_col_index,
    hipFloatComplex *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum_real = 0.f;
            float partial_sum_imag = 0.f;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                  diag_ptr = j;
                }
                partial_sum_real += fabs(hipCrealf(coo_values[j]));
                partial_sum_imag += fabs(hipCimagf(coo_values[j]));
                j++;
            }
            coo_values[diag_ptr] = make_hipFloatComplex(partial_sum_real + 1, partial_sum_imag + 1);
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_hipFloatComplex(hipCrealf(coo_values[k])/partial_sum_real, hipCimagf(coo_values[k])/partial_sum_imag);
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            float partial_sum_real = 0.f;
            float partial_sum_imag = 0.f;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum_real += fabs(hipCrealf(coo_values[j]));
                partial_sum_imag += fabs(hipCimagf(coo_values[j]));
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_hipFloatComplex(hipCrealf(coo_values[k])/partial_sum_real, hipCimagf(coo_values[k])/partial_sum_imag);
            }
        }
    }
}

void mat_adjust_nnz_z(
    const int *coo_row_index,
    const int *coo_col_index,
    hipDoubleComplex *coo_values,
    const int m,
    const int n,
    const int nnz,
    const alphasparse_fill_mode_t fillmode,
    const alphasparse_diag_type_t diagtype
) {
    if (diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum_real = 0.0;
            double partial_sum_imag = 0.0;
            int diag_ptr = j;
            while(j < nnz && row_id == coo_row_index[j]) {
                if (row_id == coo_col_index[j]) {
                  diag_ptr = j;
                }
                partial_sum_real += fabs(hipCreal(coo_values[j]));
                partial_sum_imag += fabs(hipCimag(coo_values[j]));
                j++;
            }
            coo_values[diag_ptr] = make_hipDoubleComplex(partial_sum_real + 1, partial_sum_imag + 1);
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_hipDoubleComplex(hipCreal(coo_values[k])/partial_sum_real, hipCimag(coo_values[k])/partial_sum_imag);
            }
        }
    } else {
        for (int i = 0, j = 0; i < nnz; i = j) {
            int row_id = coo_row_index[i];
            j = i;
            int col_start = j;
            double partial_sum_real = 0.0;
            double partial_sum_imag = 0.0;
            while(j < nnz && row_id == coo_row_index[j]) {
                partial_sum_real += fabs(hipCreal(coo_values[j]));
                partial_sum_imag += fabs(hipCimag(coo_values[j]));
                j++;
            }
            for (int k = col_start; k < j; k++) {
                coo_values[k] = make_hipDoubleComplex(hipCreal(coo_values[k])/partial_sum_real, hipCimag(coo_values[k])/partial_sum_imag);
            }
        }
    }
}


/*
    description:
        检查coo格式的矩阵A是否存在没有非零元的行
*/
template <typename T>
bool has_coo_zero_row(
        const T nnz,
        const T *coo_row_idx
) {
  for (T i = 1; i < nnz; i++) {
    if (coo_row_idx[i] - coo_row_idx[i - 1] > 1) {
      return true;
    }
  }
  return false;
}


/*
    description:
        检查coo格式的矩阵A是否存在有对角线元素没有非零元的情况
*/
template <typename T>
bool has_coo_zero_diag(
        const T m,
        const T nnz,
        const T *coo_row_idx,
        const T *coo_col_idx
) {
  int i = 0;
  int row;
  for (row = 0; row < m; row++) {
    while (i < nnz && coo_row_idx[i] == row && coo_row_idx[i] > coo_col_idx[i]) {
      i++;
    }
    if (i >= nnz || coo_row_idx[i] != row || coo_row_idx[i] != coo_col_idx[i]) {
      printf("i: %d\n", i);
      if (i < nnz) {
        printf("row: %d, col: %d\n", coo_row_idx[i], coo_col_idx[i]);
      }
      return true;
    }
    while (i < nnz && coo_row_idx[i] == row) {
      i++;
    }
  }
  if (!(row == m)) {
    printf("row: %d, m: %d\n", row, m);
  }
  return !(row == m);
}


template <typename T, typename U>
void mat_get_triangular_part(
    T **coo_row_idx,
    T **coo_col_idx,
    U **coo_val,
    T m,
    T n,
    T &nnz,
    alphasparse_fill_mode_t fillmode)
{
    assert(m == n);
    int k = 0;
    if (fillmode == ALPHA_SPARSE_FILL_MODE_LOWER) {
        for (int i = 0; i < nnz; i++) {
            if ((*coo_row_idx)[i] >= (*coo_col_idx)[i]) {
                (*coo_row_idx)[k] = (*coo_row_idx)[i];
                (*coo_col_idx)[k] = (*coo_col_idx)[i];
                (*coo_val)[k] = (*coo_val)[i];
                k++;
            }
        }
    } else if (fillmode == ALPHA_SPARSE_FILL_MODE_UPPER) {
        for (int i = 0; i < nnz; i++) {
            if ((*coo_row_idx)[i] <= (*coo_col_idx)[i]) {
                (*coo_row_idx)[k] = (*coo_row_idx)[i];
                (*coo_col_idx)[k] = (*coo_col_idx)[i];
                (*coo_val)[k] = (*coo_val)[i];
                k++;
            }
        }
    }
    nnz = k;
}
