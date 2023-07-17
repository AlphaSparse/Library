#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../test_common.h"
#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"
#include "alphasparse.h"
#include <iostream>

const char* file;
int thread_num;
bool check_flag;
int iter;

alphasparseOperation_t transAT;
alphasparseOperation_t transBT;

long long columns;
int C_rows, C_cols, rnnz;
int *coo_row_index, *coo_col_index;
cuFloatComplex* coo_values;

// parms for kernel
cuFloatComplex *hmatA, *hmatB, *matC_ict, *matC_cuda;
long long A_rows, A_cols;
long long B_rows, B_cols;
long long lda, ldb;
long long A_size, B_size;
const cuFloatComplex alpha = {1.1f,2.4f};
const cuFloatComplex beta = {3.2f,4.3f};

#define CHECK_CUDA(func)                                                       \
    {                                                                            \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",               \
                __LINE__,                                                         \
                cudaGetErrorString(status),                                       \
                status);                                                          \
        exit(-1);                                                                \
    }                                                                          \
    }

#define CHECK_CUSPARSE(func)                                                   \
    {                                                                            \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",           \
                __LINE__,                                                         \
                cusparseGetErrorString(status),                                   \
                status);                                                          \
        exit(-1);                                                                \
    }                                                                          \
    }

static void
cuda_sddmm()
{
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dC_offsets, *dC_columns;
    cuFloatComplex *dC_values, *dB, *dA;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size * sizeof(cuFloatComplex)) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size * sizeof(cuFloatComplex)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_offsets,
                            (C_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dC_columns, rnnz * sizeof(int))   )
    CHECK_CUDA( cudaMalloc((void**) &dC_values,  rnnz * sizeof(cuFloatComplex)) )

    CHECK_CUDA( cudaMemcpy(dA, hmatA, A_size * sizeof(cuFloatComplex),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hmatB, B_size * sizeof(cuFloatComplex),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_offsets, coo_row_index,
                            (C_rows + 1) * sizeof(int),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, coo_col_index, rnnz * sizeof(int),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, coo_values, rnnz * sizeof(cuFloatComplex),
                            cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = NULL;
    cusparseDnMatDescr_t matA, matB;
    cusparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
    // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, A_rows, A_cols, lda, dA,
                                        CUDA_C_32F, CUSPARSE_ORDER_COL) )
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                        CUDA_C_32F, CUSPARSE_ORDER_COL) )
    // Create sparse matrix C in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matC, C_rows, C_cols, rnnz,
                                        dC_offsets, dC_columns, dC_values,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F) )
    // allocate an external buffer if needed
    cusparseOperation_t transA, transB;
    if(transAT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transA = CUSPARSE_OPERATION_TRANSPOSE;
    else if(transAT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) transA = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    else transA = CUSPARSE_OPERATION_NON_TRANSPOSE;

    if(transBT == ALPHA_SPARSE_OPERATION_TRANSPOSE) transB = CUSPARSE_OPERATION_TRANSPOSE;
    else if(transBT == ALPHA_SPARSE_OPERATION_CONJUGATE_TRANSPOSE) transB = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    else transB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    printf("CUDA transA %d transB %d\n", transA, transB);
    CHECK_CUSPARSE( cusparseSDDMM_bufferSize(
                                    handle,
                                    transA,
                                    transB,
                                    &alpha, matA, matB, &beta, matC, CUDA_C_32F,
                                    CUSPARSE_SDDMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute preprocess (optional)
    CHECK_CUSPARSE( cusparseSDDMM_preprocess(
                                    handle,
                                    transA,
                                    transB,
                                    &alpha, matA, matB, &beta, matC, CUDA_C_32F,
                                    CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // execute SpMM
    CHECK_CUSPARSE( cusparseSDDMM(handle,
                                    transA,
                                    transB,
                                    &alpha, matA, matB, &beta, matC, CUDA_C_32F,
                                    CUSPARSE_SDDMM_ALG_DEFAULT, dBuffer) )
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroyDnMat(matA) )
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matC) )
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA( cudaMemcpy(matC_cuda, dC_values, rnnz * sizeof(cuFloatComplex),
                            cudaMemcpyDeviceToHost) )

    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC_offsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
}

static void
alpha_sddmm()
{
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dC_offsets, *dC_columns;
    cuFloatComplex *dC_values, *dB, *dA;
    cudaMalloc((void**) &dA, A_size * sizeof(cuFloatComplex));
    cudaMalloc((void**) &dB, B_size * sizeof(cuFloatComplex));
    cudaMalloc((void**) &dC_offsets,
                            (C_rows + 1) * sizeof(int));
    cudaMalloc((void**) &dC_columns, rnnz * sizeof(int));
    cudaMalloc((void**) &dC_values,  rnnz * sizeof(cuFloatComplex));

    cudaMemcpy(dA, hmatA, A_size * sizeof(cuFloatComplex),
                            cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hmatB, B_size * sizeof(cuFloatComplex),
                            cudaMemcpyHostToDevice);
    CHECK_CUDA( cudaMemcpy(dC_offsets, coo_row_index,
                            (C_rows + 1) * sizeof(int),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_columns, coo_col_index, rnnz * sizeof(int),
                            cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC_values, coo_values, rnnz * sizeof(cuFloatComplex),
                            cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    alphasparseHandle_t     handle = NULL;
    alphasparseDnMatDescr_t matA, matB;
    alphasparseSpMatDescr_t matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    initHandle(&handle);
    alphasparseGetHandle(&handle);
    
    // Create dense matrix A
    alphasparseCreateDnMat(&matA, A_rows, A_cols, lda, dA,
                                        ALPHA_C_32F, ALPHASPARSE_ORDER_COL);
    // Create dense matrix B
    alphasparseCreateDnMat(&matB, B_rows, B_cols, ldb, dB,
                                        ALPHA_C_32F, ALPHASPARSE_ORDER_COL);
    // Create sparse matrix C in CSR format
    alphasparseCreateCsr(&matC, C_rows, C_cols, rnnz,
                                        dC_offsets, dC_columns, dC_values,
                                        ALPHA_SPARSE_INDEXTYPE_I32, ALPHA_SPARSE_INDEXTYPE_I32,
                                        ALPHA_SPARSE_INDEX_BASE_ZERO, ALPHA_C_32F);
    // allocate an external buffer if needed
    printf("ALPHA transA %d transB %d\n", transAT, transBT);
    alphasparseSDDMM_bufferSize(handle,
                            transAT,
                            transBT,
                            &alpha, matA, matB, &beta, matC, ALPHA_C_32F,
                            ALPHASPARSE_SDDMM_ALG_DEFAULT, &bufferSize);
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // execute preprocess (optional)
    alphasparseSDDMM_preprocess(handle,
                            transAT,
                            transBT,
                            &alpha, matA, matB, &beta, matC, ALPHA_C_32F,
                            ALPHASPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    CHECK_CUDA( cudaDeviceSynchronize() )
    // execute SpMM
    alphasparseSDDMM(handle,
                transAT,
                transBT,
                &alpha, matA, matB, &beta, matC, ALPHA_C_32F,
                ALPHASPARSE_SDDMM_ALG_DEFAULT, dBuffer);
    CHECK_CUDA( cudaDeviceSynchronize() )
    // destroy matrix/vector descriptors
    // cusparseDestroyDnMat(matA);
    // cusparseDestroyDnMat(matB);
    // cusparseDestroySpMat(matC);
    // alphasparseDestroy(handle);
    //--------------------------------------------------------------------------
    // device result check
    CHECK_CUDA(cudaMemcpy(matC_ict, dC_values, rnnz * sizeof(cuFloatComplex),  cudaMemcpyDeviceToHost) )

    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC_offsets) )
    CHECK_CUDA( cudaFree(dC_columns) )
    CHECK_CUDA( cudaFree(dC_values) )
}

int main(int argc, const char *argv[]) {
    // Host problem definition
    args_help(argc, argv);
    file = args_get_data_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    transAT = alpha_args_get_transA(argc, argv);
    transBT = alpha_args_get_transB(argc, argv);

    // read coo
    alpha_read_coo<cuFloatComplex>(
    file, &C_rows, &C_cols, &rnnz, &coo_row_index, &coo_col_index, &coo_values);
    coo_order<int32_t, cuFloatComplex>(rnnz, coo_row_index, coo_col_index, coo_values);
    columns = args_get_cols(argc, argv, C_rows); // 默认C是方阵
    A_rows = C_rows;
    A_cols = columns;
    B_rows = A_cols;
    B_cols = C_cols;
    lda = A_rows;
    ldb = B_rows;

    A_size = lda * A_cols;
    B_size = ldb * B_cols;
    
    for (int i = 0; i < 20; i++) {
    std::cout << coo_row_index[i] << ", ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 20; i++) {
    std::cout << coo_col_index[i] << ", ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 20; i++) {
    std::cout << coo_values[i] << ", ";
    }
    std::cout << std::endl;
    // init x y
    // init B C
    hmatA = (cuFloatComplex*)alpha_malloc(A_size * sizeof(cuFloatComplex));
    hmatB = (cuFloatComplex*)alpha_malloc(B_size * sizeof(cuFloatComplex));
    matC_ict = (cuFloatComplex*)alpha_malloc(rnnz * sizeof(cuFloatComplex));
    matC_cuda = (cuFloatComplex*)alpha_malloc(rnnz * sizeof(cuFloatComplex));

    alpha_fill_random(hmatA, 1, A_size);
    alpha_fill_random(hmatB, 1, B_size);
    memset(matC_ict, 0, rnnz * sizeof(cuFloatComplex));
    memset(matC_cuda, 0, rnnz * sizeof(cuFloatComplex));

    cuda_sddmm();
    alpha_sddmm();
    
    for (int i = 0; i < 20; i++) {
    std::cout << matC_cuda[i] << ", ";
    }
    std::cout << std::endl;
    for (int i = 0; i < 20; i++) {
        std::cout << matC_ict[i] << ", ";
    }
    check((cuFloatComplex*)matC_cuda, rnnz, (cuFloatComplex*)matC_ict, rnnz);
    
    return EXIT_SUCCESS;
}