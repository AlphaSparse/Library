#include "../test_common.h"

#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_csr.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"

#include "../include/spsv/mat_adjust.h"
#include "../include/common.h"
#include "alphasparse.h"
#include <iostream>
#include <chrono>

typedef double DATA_TYPE;

const char* file;
const char* metrics_file;
bool check_flag;
bool metrics_flag;
int iter = 1;
int warmup = 1;

const char* filename;

float cuda_time_bufferSize;
float cuda_time_analysis;
float cuda_time_solve_avg;
double cuda_time_total;

float alpha_time_bufferSize;
float alpha_time_analysis = 0.f;
float alpha_time_solve_avg;
double alpha_time_total;

double speedup;

// int m, n;
int nnz;
int *coo_row_index, *coo_col_index;
DATA_TYPE* coo_values;

int A_rows, A_cols, rnnz;
long long n_rhs;
long long C_rows, C_cols;
long long B_rows, B_cols;
long long ldb, ldc;
long long B_size, C_size;

DATA_TYPE* hmatB;
DATA_TYPE* hmatC_cuda;
DATA_TYPE* hmatC_alpha;

double error;

const DATA_TYPE alpha = 2.f;

// parms for kernel
hipDataType cuda_datatype;
alphasparseDataType alpha_datatype;

hipsparseOrder_t cuda_order;
alphasparseOrder_t alpha_order;

hipsparseOperation_t cuda_opA, cuda_opB;
alphasparseOperation_t alpha_opA, alpha_opB;

hipsparseFillMode_t cuda_fillmode;
alphasparse_fill_mode_t alpha_fillmode;

hipsparseDiagType_t cuda_diagtype;
alphasparse_diag_type_t alpha_diagtype;

hipsparseSpSMAlg_t cuda_alg;
alphasparseSpSMAlg_t alpha_alg;

hipEvent_t event_start, event_stop;

static void
cuda_spsm() {
    hipsparseHandle_t handle = NULL;
    hipsparseCreate(&handle);
    // Offload data to device
    DATA_TYPE* dmatB = NULL;
    DATA_TYPE* dmatC = NULL;
    int* dCsrRowPtr = NULL;
    int* dArow = NULL;
    int* dAcol = NULL;
    DATA_TYPE* dAval = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(DATA_TYPE) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
    CHECK_CUDA(hipMemcpy(dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAval, coo_values, nnz * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);
    CHECK_CUDA(hipMalloc((void**)&dmatB, sizeof(DATA_TYPE) * B_size));
    CHECK_CUDA(hipMalloc((void**)&dmatC, sizeof(DATA_TYPE) * C_size));
    CHECK_CUDA(hipMemcpy(dmatB, hmatB, sizeof(DATA_TYPE) * B_size, hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dmatC, hmatC_cuda, sizeof(DATA_TYPE) * C_size, hipMemcpyHostToDevice));
    hipsparseDnMatDescr_t matB, matC;
    hipsparseSpMatDescr_t matA;
    (hipsparseCreateDnMat(&matB, A_cols, B_cols, ldb, dmatB, cuda_datatype, cuda_order));
    (hipsparseCreateDnMat(&matC, C_rows, C_cols, ldc, dmatC, cuda_datatype, cuda_order));
    (hipsparseCreateCsr(&matA,
                                    A_rows,
                                    A_cols,
                                    nnz,
                                    dCsrRowPtr,
                                    dAcol,
                                    dAval,
                                    HIPSPARSE_INDEX_32I,
                                    HIPSPARSE_INDEX_32I,
                                    HIPSPARSE_INDEX_BASE_ZERO,
                                    cuda_datatype));
    hipsparseSpSMDescr_t spsmDescr;
    (hipsparseSpSM_createDescr(&spsmDescr));
    (hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_FILL_MODE, &cuda_fillmode, sizeof(cuda_fillmode)));
    (hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_DIAG_TYPE, &cuda_diagtype, sizeof(cuda_diagtype)));
    void* dBuffer = NULL;
    size_t bufferSize = 0;

    GPU_TIMER_START(cuda_time_bufferSize, event_start, event_stop);
    (hipsparseSpSM_bufferSize(handle,
                                            cuda_opA,
                                            cuda_opB,
                                            &alpha,
                                            matA,
                                            matB,
                                            matC,
                                            cuda_datatype,
                                            cuda_alg,
                                            spsmDescr,
                                            &bufferSize));
    GPU_TIMER_END(cuda_time_bufferSize, event_start, event_stop);

    CHECK_CUDA(hipMalloc(&dBuffer, bufferSize));

    GPU_TIMER_START(cuda_time_analysis, event_start, event_stop);
    (hipsparseSpSM_analysis(handle,
                                        cuda_opA,
                                        cuda_opB,
                                        &alpha,
                                        matA,
                                        matB,
                                        matC,
                                        cuda_datatype,
                                        cuda_alg,
                                        spsmDescr,
                                        dBuffer));
    GPU_TIMER_END(cuda_time_analysis, event_start, event_stop);

    std::vector<double> times_solve;
    for (int i = 0; i < iter + warmup; i++) {
        float time_solve;
        GPU_TIMER_START(time_solve, event_start, event_stop);
        (hipsparseSpSM_solve(handle,
                                            cuda_opA,
                                            cuda_opB,
                                            &alpha,
                                            matA,
                                            matB,
                                            matC,
                                            cuda_datatype,
                                            cuda_alg,
                                            spsmDescr,
                                            dBuffer));
        hipDeviceSynchronize();
        GPU_TIMER_END(time_solve, event_start, event_stop);
        if (i >= warmup) {
            times_solve.push_back(time_solve);
        }
    }
    cuda_time_solve_avg = get_avg_time_2(times_solve);

    CHECK_CUDA(hipMemcpy(hmatC_cuda, dmatC, C_size * sizeof(DATA_TYPE), hipMemcpyDeviceToHost));
    
    (hipsparseDestroySpMat(matA));
    (hipsparseDestroyDnMat(matB));
    (hipsparseDestroyDnMat(matC));
    (hipsparseSpSM_destroyDescr(spsmDescr));
    (hipsparseDestroy(handle));
    // Clear up on device
    CHECK_CUDA(hipFree(dArow));
    CHECK_CUDA(hipFree(dAcol));
    CHECK_CUDA(hipFree(dAval));
    CHECK_CUDA(hipFree(dCsrRowPtr));
    CHECK_CUDA(hipFree(dmatB));
    CHECK_CUDA(hipFree(dmatC));
    CHECK_CUDA(hipFree(dBuffer));
}


static void
alpha_spsm() {
    alphasparseHandle_t handle = NULL;
    initHandle(&handle);
    alphasparseGetHandle(&handle);
    // Offload data to device
    DATA_TYPE* dmatB = NULL;
    DATA_TYPE* dmatC = NULL;
    int* dCsrRowPtr = NULL;
    int* dArow = NULL;
    int* dAcol = NULL;
    DATA_TYPE* dAval = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dArow, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAcol, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dAval, sizeof(DATA_TYPE) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void**)&dCsrRowPtr, sizeof(int) * (A_rows + 1)));
    CHECK_CUDA(hipMemcpy(dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAval, coo_values, nnz * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    alphasparseXcoo2csr(dArow, nnz, A_rows, dCsrRowPtr);
    alphasparseDnMatDescr_t matB, matC;
    alphasparseSpMatDescr_t matA;
    CHECK_CUDA(hipMalloc((void**)&dmatB, sizeof(DATA_TYPE) * B_size));
    CHECK_CUDA(hipMalloc((void**)&dmatC, sizeof(DATA_TYPE) * C_size));
    CHECK_CUDA(hipMemcpy(dmatB, hmatB, sizeof(DATA_TYPE) * B_size, hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dmatC, hmatC_alpha, sizeof(DATA_TYPE) * C_size, hipMemcpyHostToDevice));
    alphasparseCreateDnMat(&matB, A_cols, B_cols, ldb, dmatB, alpha_datatype, alpha_order);
    alphasparseCreateDnMat(&matC, C_rows, C_cols, ldc, dmatC, alpha_datatype, alpha_order);
    alphasparseCreateCsr(&matA,
                        A_rows,
                        A_cols,
                        nnz,
                        dCsrRowPtr,
                        dAcol,
                        dAval,
                        ALPHA_SPARSE_INDEXTYPE_I32,
                        ALPHA_SPARSE_INDEXTYPE_I32,
                        ALPHA_SPARSE_INDEX_BASE_ZERO,
                        alpha_datatype);
    alphasparseSpSMDescr_t spsmDescr;
    alphasparseSpSM_createDescr(&spsmDescr);
    alphasparseSpMatSetAttribute(matA, ALPHASPARSE_SPMAT_FILL_MODE, &alpha_fillmode, sizeof(alpha_fillmode));
    alphasparseSpMatSetAttribute(matA, ALPHASPARSE_SPMAT_DIAG_TYPE, &alpha_diagtype, sizeof(alpha_diagtype));
    size_t bufferSize = 0;
    void* dBuffer = NULL;
    
    GPU_TIMER_START(alpha_time_bufferSize, event_start, event_stop);
    alphasparseSpSM_bufferSize(handle,
                                alpha_opA,
                                alpha_opB,
                                &alpha,
                                matA,
                                matB,
                                matC,
                                alpha_datatype,
                                alpha_alg,
                                spsmDescr,
                                &bufferSize);
    GPU_TIMER_END(alpha_time_bufferSize, event_start, event_stop);

    CHECK_CUDA(hipMalloc(&dBuffer, bufferSize))

    // GPU_TIMER_START(alpha_time_analysis, event_start, event_stop);
    // GPU_TIMER_END(alpha_time_analysis, event_start, event_stop);

    std::vector<double> times_solve;
    for (int i = 0; i < iter + warmup; i++) {
        float time_solve;
        GPU_TIMER_START(time_solve, event_start, event_stop);
        alphasparseSpSM_solve(handle,
                                alpha_opA,
                                alpha_opB,
                                &alpha,
                                matA,
                                matB,
                                matC,
                                alpha_datatype,
                                alpha_alg,
                                spsmDescr);
        GPU_TIMER_END(time_solve, event_start, event_stop);
        if (i >= warmup) {
            times_solve.push_back(time_solve);
        }
    }
    alpha_time_solve_avg = get_avg_time_2(times_solve);

    CHECK_CUDA(hipMemcpy(hmatC_alpha, dmatC, C_size * sizeof(DATA_TYPE), hipMemcpyDeviceToHost))
    // Clear up on device
    CHECK_CUDA(hipFree(dArow));
    CHECK_CUDA(hipFree(dAcol));
    CHECK_CUDA(hipFree(dAval));
    CHECK_CUDA(hipFree(dCsrRowPtr));
    CHECK_CUDA(hipFree(dmatB));
    CHECK_CUDA(hipFree(dmatC));
    CHECK_CUDA(hipFree(dBuffer));
}

int 
main(int argc, const char* argv[]) {
    args_help_hip_hip(argc, argv);
    file = args_get_data_file(argc, argv);
    metrics_file = args_save_metrics_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    metrics_flag = args_get_if_calculate_metrics(argc, argv);
    iter = args_get_iter(argc, argv);
    warmup = args_get_warmup(argc, argv);

    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    alphasparseOperation_t transB = alpha_args_get_transB(argc, argv);
    alpha_matrix_descr descrT = alpha_args_get_matrix_descrA(argc, argv);

    int algo_num = args_get_alg_num(argc, argv);

    alpha_datatype = get_alpha_datatype<DATA_TYPE>();
    cuda_datatype = alpha2cuda_datatype_map[alpha_datatype];
    alpha_opA = transA;
    cuda_opA = alpha2cuda_op_map[alpha_opA];
    alpha_opB = transB;
    cuda_opB = alpha2cuda_op_map[alpha_opB];
    alpha_fillmode = descrT.mode;
    cuda_fillmode = alpha2cuda_fill_map[alpha_fillmode];
    alpha_diagtype = descrT.diag;
    cuda_diagtype = alpha2cuda_diag_map[alpha_diagtype];
    alpha_order = alpha_args_get_order(argc, argv);
    cuda_order = alpha2cuda_order_map[alpha_order];
    alpha_alg = get_alpha_spsm_alg(algo_num);
    cuda_alg = HIPSPARSE_SPSM_ALG_DEFAULT;

    filename = get_filename(file);

    // read coo
    alpha_read_coo<DATA_TYPE>(file, &A_rows, &A_cols, &nnz, &coo_row_index, &coo_col_index, &coo_values);
    if (A_rows != A_cols || A_rows == 0) {
        return 0;
    }
    if (nnz < 1e+6) {
        return 0;
    }
    coo_order<int32_t, DATA_TYPE>(nnz, coo_row_index, coo_col_index, coo_values);
    if (alpha_diagtype == ALPHA_SPARSE_DIAG_NON_UNIT) {
        // 补充对角线元素
        mat_patch_trim_d<int32_t>(&A_rows, &A_cols, &nnz, &coo_row_index, &coo_col_index, &coo_values);
    }
    // 矩阵行元素求和归一化
    mat_adjust_nnz_d(coo_row_index, coo_col_index, coo_values, A_rows, A_cols, nnz, alpha_fillmode, alpha_diagtype);

    n_rhs = 32; // 默认C是方阵
    B_rows = A_cols;
    B_cols = n_rhs;
    C_rows = A_rows;
    C_cols = B_cols;
    if (alpha_order == ALPHASPARSE_ORDER_ROW) {
        ldb = B_cols;
        ldc = C_cols;
        B_size = ldb * B_rows;
        C_size = ldc * C_rows;
    } else if (alpha_order == ALPHASPARSE_ORDER_COL) {
        ldb = B_rows;
        ldc = C_rows;
        B_size = ldb * B_cols;
        C_size = ldc * C_cols;
    }
    
    hmatB = (DATA_TYPE*)alpha_malloc(B_size * sizeof(DATA_TYPE));
    
    alpha_fill_random(hmatB, 0, B_size);
    
    int execute_type = 1;
    if (execute_type == 0) {
        if (algo_num < 0) {
            // printf("\n%s,", filename);
            hmatC_cuda = (DATA_TYPE*)alpha_malloc(C_size * sizeof(DATA_TYPE));
            alpha_fill_random(hmatC_cuda, 1, C_size);
            cuda_spsm();
            cuda_time_total = cuda_time_bufferSize + cuda_time_analysis + cuda_time_solve_avg;
            if (metrics_flag) {
                std::ofstream outfile(metrics_file, std::ios::app);
                outfile << "CSR,"
                        << cuda_op_map[cuda_opA] << ","
                        << cuda_op_map[cuda_opB] << ","
                        << cuda_fill_map[cuda_fillmode] << ","
                        << cuda_diag_map[cuda_diagtype] << ","
                        << cuda_order_map[cuda_order] << ","
                        << cuda_spsm_alg_map[cuda_alg] << ","
                        << cuda_datatype_map[cuda_datatype] << ","
                        << filename << ","
                        << cuda_time_bufferSize << ","
                        << cuda_time_analysis << ","
                        << cuda_time_solve_avg << ","
                        << cuda_time_total << "\n";
                outfile.close();
            }
            printf("%.6lf,", cuda_time_total);
            free(hmatC_cuda);
        } else {
            hmatC_alpha = (DATA_TYPE*)alpha_malloc(C_size * sizeof(DATA_TYPE));
            alpha_fill_random(hmatC_alpha, 1, C_size);
            alpha_spsm();
            alpha_time_total = alpha_time_bufferSize + alpha_time_analysis + alpha_time_solve_avg;
            if (metrics_flag) {
                std::ofstream outfile(metrics_file, std::ios::app);
                outfile << "CSR,"
                        << alpha_op_map[alpha_opA] << ","
                        << alpha_op_map[alpha_opB] << ","
                        << alpha_fill_map[alpha_fillmode] << ","
                        << alpha_diag_map[alpha_diagtype] << ","
                        << alpha_order_map[alpha_order] << ","
                        << alpha_spsm_alg_map[alpha_alg] << ","
                        << alpha_datatype_map[alpha_datatype] << ","
                        << filename << ","
                        << alpha_time_bufferSize << ","
                        << alpha_time_analysis << ","
                        << alpha_time_solve_avg << ","
                        << alpha_time_total << "\n";
                outfile.close();
            }
            printf("%.6lf,", alpha_time_total);
            free(hmatC_alpha);
        }
    } else if (execute_type == 1) {
        hmatC_cuda = (DATA_TYPE*)alpha_malloc(C_size * sizeof(DATA_TYPE));
        alpha_fill_random(hmatC_cuda, 1, C_size);
        cuda_spsm();
        cuda_time_total = cuda_time_bufferSize + cuda_time_analysis + cuda_time_solve_avg;
        hmatC_alpha = (DATA_TYPE*)alpha_malloc(C_size * sizeof(DATA_TYPE));
        alpha_fill_random(hmatC_alpha, 1, C_size);
        alpha_spsm();
        alpha_time_total = alpha_time_bufferSize + alpha_time_analysis + alpha_time_solve_avg;
        speedup = cuda_time_total / alpha_time_total;
        if (check_flag) {
            check((DATA_TYPE*)hmatC_cuda, C_size, (DATA_TYPE*)hmatC_alpha, C_size, &error);
        }
        if (metrics_flag) {
            std::ofstream outfile(metrics_file, std::ios::app);
            outfile << "CSR,"
                    << cuda_op_map[cuda_opA] << ","
                    << cuda_op_map[cuda_opB] << ","
                    << cuda_fill_map[cuda_fillmode] << ","
                    << cuda_diag_map[cuda_diagtype] << ","
                    << cuda_order_map[cuda_order] << ","
                    << cuda_spsm_alg_map[cuda_alg] << ","
                    << cuda_datatype_map[cuda_datatype] << ","
                    << filename << ","
                    << cuda_time_bufferSize << ","
                    << cuda_time_analysis << ","
                    << cuda_time_solve_avg << ","
                    << cuda_time_total << "\n";
            outfile << "CSR,"
                    << alpha_op_map[alpha_opA] << ","
                    << alpha_op_map[alpha_opB] << ","
                    << alpha_fill_map[alpha_fillmode] << ","
                    << alpha_diag_map[alpha_diagtype] << ","
                    << alpha_order_map[alpha_order] << ","
                    << alpha_spsm_alg_map[alpha_alg] << ","
                    << alpha_datatype_map[alpha_datatype] << ","
                    << filename << ","
                    << alpha_time_bufferSize << ","
                    << alpha_time_analysis << ","
                    << alpha_time_solve_avg << ","
                    << alpha_time_total << ","
                    << speedup;
            if (check_flag) {
                outfile << "," << error;
            }
            outfile << "\n";
            outfile.close();
        }
        // printf("%s,", filename);
        printf("%.6lf,", speedup);

        // if (check_flag) {
        //     std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
        //     std::cout.precision(12);
        //     std::cout << "\nError: " << error << std::endl;
        //     std::cout << "\t i\t\tcuda_y, \t\tict_y\n"; 

        //     for (int i = 0, cnt = 0; i < C_size && cnt < 20; i++) {
        //         if (fabs(hmatC_cuda[i] - hmatC_alpha[i]) > 1e-10) {
        //             std::cout << cnt << ", \t";
        //             std::cout << i << ", \t";
        //             std::cout << hmatC_cuda[i] << ", \t";
        //             std::cout << hmatC_alpha[i] << std::endl;
        //             cnt++;
        //         }
        //     }
        //     std::cout << std::endl;
        // }

        free(hmatC_cuda);
        free(hmatC_alpha);
    }
    free(hmatB);

    // // print speedup info
    // // 获取当前时间
    // auto currentTime = std::chrono::system_clock::now();
    // std::time_t currentTimeT = std::chrono::system_clock::to_time_t(currentTime);
    // // 转换为本地时间结构
    // std::tm localTime = *std::localtime(&currentTimeT);
    // // 格式化时间为字符串
    // std::ostringstream timeString;
    // timeString << std::put_time(&localTime, "%Y/%m/%d-%H:%M:%S");
    // std::cout << timeString.str() << ",";
    // printf("%s,%.6lf\n", filename, speedup);

    return 0;
}