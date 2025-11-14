#include "../test_common.h"

#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <vector>

#include "../../format/alphasparse_create_coo.h"
#include "../../format/coo2csr.h"
#include "../../format/coo_order.h"

#include "../include/spsv/mat_adjust.h"
#include "../include/common.h"
#include "alphasparse.h"
#include <iostream>
#include <chrono>

typedef float DATA_TYPE;

const char *file;
const char *metrics_file;
int thread_num;
bool check_flag;
bool metrics_flag;
int iter = 1;
int warmup = 1;

const char *filename;

float cuda_time_bufferSize;
float cuda_time_analysis;
float cuda_time_solve_avg;
double cuda_time_total;
// double cuda_bandwidth;
// double cuda_gflops;

float alpha_time_bufferSize;
float alpha_time_analysis;
float alpha_time_solve_avg;
double alpha_time_total;
// double alpha_bandwidth;
// double alpha_gflops;

double speedup;

int m, n, nnz;
int *coo_row_index, *coo_col_index;
DATA_TYPE *coo_values;

// coo format
DATA_TYPE *x_val;
DATA_TYPE *ict_y;
DATA_TYPE *cuda_y;

float error;

// parms for kernel
const DATA_TYPE alpha = 2.f;

hipDataType cuda_datatype;
alphasparseDataType alpha_datatype;

hipsparseOperation_t cuda_op;
alphasparseOperation_t alpha_op;

hipsparseFillMode_t cuda_fillmode;
alphasparse_fill_mode_t alpha_fillmode;

hipsparseDiagType_t cuda_diagtype;
alphasparse_diag_type_t alpha_diagtype;

hipsparseSpSVAlg_t cuda_alg;
alphasparseSpSVAlg_t alpha_alg;

hipEvent_t event_start, event_stop;

static void
cuda_spsv()
{
    hipsparseHandle_t handle = NULL;
    (hipsparseCreate(&handle));

    // Offload data to device
    DATA_TYPE *dX = NULL;
    DATA_TYPE *dY = NULL;
    int *dArow = NULL;
    int *dAcol = NULL;
    DATA_TYPE *dAval = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dArow, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(DATA_TYPE) * nnz));
    CHECK_CUDA(hipMemcpy(dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAval, coo_values, nnz * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    hipsparseDnVecDescr_t vecX, vecY;
    hipsparseSpMatDescr_t matA;
    CHECK_CUDA(hipMalloc((void **)&dX, n * sizeof(DATA_TYPE)));
    CHECK_CUDA(hipMalloc((void **)&dY, m * sizeof(DATA_TYPE)));
    CHECK_CUDA(hipMemcpy(dX, x_val, n * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dY, cuda_y, m * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    // Create dense vector X
    (hipsparseCreateDnVec(&vecX, n, dX, cuda_datatype));
    // Create dense vector y
    (hipsparseCreateDnVec(&vecY, m, dY, cuda_datatype));
    (hipsparseCreateCoo(&matA,
                        m,
                        n,
                        nnz,
                        dArow,
                        dAcol,
                        dAval,
                        HIPSPARSE_INDEX_32I,
                        HIPSPARSE_INDEX_BASE_ZERO,
                        cuda_datatype));
    hipsparseSpSVDescr_t spsvDescr;
    hipsparseSpSV_createDescr(&spsvDescr);
    (hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_FILL_MODE, &cuda_fillmode, sizeof(cuda_fillmode)));
    (hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_DIAG_TYPE, &cuda_diagtype, sizeof(cuda_diagtype)));
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // ****
    int *tmp;
    hipMalloc((void **)&tmp, sizeof(int));
    // ****

    GPU_TIMER_START(cuda_time_bufferSize, event_start, event_stop);
    (hipsparseSpSV_bufferSize(handle,
                              cuda_op,
                              &alpha,
                              matA,
                              vecX,
                              vecY,
                              cuda_datatype,
                              cuda_alg,
                              spsvDescr,
                              &bufferSize));
    GPU_TIMER_END(cuda_time_bufferSize, event_start, event_stop);

    CHECK_CUDA(hipMalloc(&dBuffer, bufferSize))

    std::vector<double> times_analysis;
    // printf("time_analysis\n");
    for (int i = 0; i < iter + warmup; i++)
    {
        float time_analysis;
        GPU_TIMER_START(time_analysis, event_start, event_stop);
        (hipsparseSpSV_analysis(handle,
                                cuda_op,
                                &alpha,
                                matA,
                                vecX,
                                vecY,
                                cuda_datatype,
                                cuda_alg,
                                spsvDescr,
                                dBuffer));
        // hipDeviceSynchronize();
        GPU_TIMER_END(time_analysis, event_start, event_stop);
        // printf("%.4lf\n", time_analysis);
        if (i >= warmup)
        {
            times_analysis.push_back(time_analysis);
        }
    }
    cuda_time_analysis = get_avg_time_2(times_analysis);

    std::vector<double> times_solve;
    for (int i = 0; i < iter + warmup; i++)
    {
        float time_solve;
        GPU_TIMER_START(time_solve, event_start, event_stop);
        (hipsparseSpSV_solve(handle,
                             cuda_op,
                             &alpha,
                             matA,
                             vecX,
                             vecY,
                             cuda_datatype,
                             cuda_alg,
                             spsvDescr));
        // hipDeviceSynchronize();
        GPU_TIMER_END(time_solve, event_start, event_stop);
        // printf("%.4lf\n", time_solve);
        if (i >= warmup)
        {
            times_solve.push_back(time_solve);
        }
    }
    cuda_time_solve_avg = get_avg_time_2(times_solve);

    CHECK_CUDA(hipMemcpy(cuda_y, dY, sizeof(DATA_TYPE) * m, hipMemcpyDeviceToHost));

    // destroy matrix/vector descriptors
    (hipsparseDestroySpMat(matA));
    (hipsparseDestroyDnVec(vecX));
    (hipsparseDestroyDnVec(vecY));
    (hipsparseSpSV_destroyDescr(spsvDescr));
    (hipsparseDestroy(handle));

    CHECK_CUDA(hipFree(dArow));
    CHECK_CUDA(hipFree(dAcol));
    CHECK_CUDA(hipFree(dAval));
    CHECK_CUDA(hipFree(dX));
    CHECK_CUDA(hipFree(dY));
    CHECK_CUDA(hipFree(dBuffer));
}

static void
alpha_spsv()
{
    alphasparseHandle_t handle;
    initHandle(&handle);
    alphasparseGetHandle(&handle);

    // Offload data to device
    DATA_TYPE *dX = NULL;
    DATA_TYPE *dY = NULL;
    int *dArow = NULL;
    int *dAcol = NULL;
    DATA_TYPE *dAval = NULL;

    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dArow, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAcol, sizeof(int) * nnz));
    PRINT_IF_HIP_ERROR(hipMalloc((void **)&dAval, sizeof(DATA_TYPE) * nnz));

    CHECK_CUDA(hipMemcpy(dArow, coo_row_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAcol, coo_col_index, nnz * sizeof(int), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dAval, coo_values, nnz * sizeof(DATA_TYPE), hipMemcpyHostToDevice));

    alphasparseDnVecDescr_t vecX, vecY;
    alphasparseSpMatDescr_t matA;
    CHECK_CUDA(hipMalloc((void **)&dX, n * sizeof(DATA_TYPE)));
    CHECK_CUDA(hipMalloc((void **)&dY, m * sizeof(DATA_TYPE)));
    CHECK_CUDA(hipMemcpy(dX, x_val, n * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    CHECK_CUDA(hipMemcpy(dY, ict_y, m * sizeof(DATA_TYPE), hipMemcpyHostToDevice));
    alphasparseCreateDnVec(&vecX, n, (void *)dX, alpha_datatype);
    alphasparseCreateDnVec(&vecY, m, (void *)dY, alpha_datatype);
    alphasparseCreateCoo(&matA,
                         m,
                         n,
                         nnz,
                         dArow,
                         dAcol,
                         dAval,
                         ALPHA_SPARSE_INDEXTYPE_I32,
                         ALPHA_SPARSE_INDEX_BASE_ZERO,
                         alpha_datatype);
    alphasparseSpSVDescr_t spsvDescr;
    alphasparseSpSV_createDescr(&spsvDescr);
    alphasparseSpMatSetAttribute(matA, ALPHASPARSE_SPMAT_FILL_MODE, &alpha_fillmode, sizeof(alpha_fillmode));
    alphasparseSpMatSetAttribute(matA, ALPHASPARSE_SPMAT_DIAG_TYPE, &alpha_diagtype, sizeof(alpha_diagtype));
    void *dBuffer = NULL;
    size_t bufferSize = 0;

    // ****
    int *tmp;
    hipMalloc((void **)&tmp, sizeof(int));
    // ****

    GPU_TIMER_START(alpha_time_bufferSize, event_start, event_stop);
    alphasparseSpSV_bufferSize(handle,
                               alpha_op,
                               &alpha,
                               matA,
                               vecX,
                               vecY,
                               alpha_datatype,
                               alpha_alg,
                               spsvDescr,
                               &bufferSize);
    GPU_TIMER_END(alpha_time_bufferSize, event_start, event_stop);

    CHECK_CUDA(hipMalloc(&dBuffer, bufferSize))

    GPU_TIMER_START(alpha_time_analysis, event_start, event_stop);
    
    std::vector<double> times_analysis;
    // printf("time_analysis\n");
    for (int i = 0; i < iter + warmup; i++)
    {
        float time_analysis;
        GPU_TIMER_START(time_analysis, event_start, event_stop);
        alphasparseSpSV_analysis(handle,
                                 alpha_op,
                                 &alpha,
                                 matA,
                                 vecX,
                                 vecY,
                                 alpha_datatype,
                                 alpha_alg,
                                 spsvDescr,
                                 dBuffer);
        // hipDeviceSynchronize();
        GPU_TIMER_END(time_analysis, event_start, event_stop);
        // printf("%.4lf\n", time_analysis);
        if (i >= warmup)
        {
            times_analysis.push_back(time_analysis);
        }
    }
    alpha_time_analysis = get_avg_time_2(times_analysis);

    std::vector<double> times_solve;
    for (int i = 0; i < iter + warmup; i++)
    {
        float time_solve;
        GPU_TIMER_START(time_solve, event_start, event_stop);
        alphasparseSpSV_solve(handle,
                              alpha_op,
                              &alpha,
                              matA,
                              vecX,
                              vecY,
                              alpha_datatype,
                              alpha_alg,
                              spsvDescr,
                              dBuffer);
        // hipDeviceSynchronize();
        GPU_TIMER_END(time_solve, event_start, event_stop);
        // printf("%.4lf\n", time_solve);
        if (i >= warmup)
        {
            times_solve.push_back(time_solve);
        }
    }
    alpha_time_solve_avg = get_avg_time_2(times_solve);
    // alpha_bandwidth = static_cast<double>(sizeof(float)) * (2 * m + nnz) +
    //                   sizeof(int) * (m + 1 + nnz) / alpha_time_solve_avg / 1e6;
    // alpha_gflops = static_cast<double>(2 * nnz) / alpha_time_solve_avg / 1e6;

    CHECK_CUDA(hipMemcpy(ict_y, dY, sizeof(DATA_TYPE) * m, hipMemcpyDeviceToHost));

    CHECK_CUDA(hipFree(dArow));
    CHECK_CUDA(hipFree(dAcol));
    CHECK_CUDA(hipFree(dAval));
    CHECK_CUDA(hipFree(dX));
    CHECK_CUDA(hipFree(dY));
    CHECK_CUDA(hipFree(dBuffer));
}

int main(int argc, const char *argv[])
{
    // args
    args_help_hip(argc, argv);
    file = args_get_data_file(argc, argv);
    metrics_file = args_save_metrics_file(argc, argv);
    check_flag = args_get_if_check(argc, argv);
    metrics_flag = args_get_if_calculate_metrics(argc, argv);
    iter = args_get_iter(argc, argv);
    warmup = args_get_warmup(argc, argv);

    alphasparseOperation_t transA = alpha_args_get_transA(argc, argv);
    alpha_matrix_descr mat_descrA = alpha_args_get_matrix_descrA(argc, argv);
    alphasparse_fill_mode_t fillA = mat_descrA.mode;
    alphasparse_diag_type_t diagA = mat_descrA.diag;

    int algo_num = args_get_alg_num(argc, argv);

    alpha_datatype = get_alpha_datatype<DATA_TYPE>();
    cuda_datatype = alpha2cuda_datatype_map[alpha_datatype];
    alpha_op = transA;
    cuda_op = alpha2cuda_op_map[alpha_op];
    alpha_fillmode = fillA;
    cuda_fillmode = alpha2cuda_fill_map[alpha_fillmode];
    alpha_diagtype = diagA;
    cuda_diagtype = alpha2cuda_diag_map[alpha_diagtype];
    alpha_alg = get_alpha_spsv_alg(algo_num);
    cuda_alg = HIPSPARSE_SPSV_ALG_DEFAULT;

    filename = get_filename(file);

    // read coo
    alpha_read_coo<DATA_TYPE>(file, &m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);

    if (m != n || m == 0)
    {
        // printf("%s - the row number and the column number are NOT equal!\n", file);
        return 0;
    }
    // if (nnz < 1e+6)
    // {
    //     return 0;
    // }

    coo_order<int32_t, DATA_TYPE>(nnz, coo_row_index, coo_col_index, coo_values);

    if (alpha_diagtype == ALPHA_SPARSE_DIAG_NON_UNIT)
    {
        // 补充对角线元素
        mat_patch_trim_s<int32_t>(&m, &n, &nnz, &coo_row_index, &coo_col_index, &coo_values);
    }
    // 矩阵行元素求和归一化
    mat_adjust_nnz_s(coo_row_index, coo_col_index, coo_values, m, n, nnz, alpha_fillmode, alpha_diagtype);

    // init x y
    x_val = (DATA_TYPE *)alpha_malloc(n * sizeof(DATA_TYPE));

    alpha_fill_random(x_val, 0, n);

    int execute_type = 1;
    if (execute_type == 0)
    {
        if (algo_num < 0)
        {
            // printf("\n%s,", filename);
            cuda_y = (DATA_TYPE *)alpha_malloc(m * sizeof(DATA_TYPE));
            alpha_fill_random(cuda_y, 1, m);
            cuda_spsv();
            cuda_time_total = cuda_time_bufferSize + cuda_time_analysis + cuda_time_solve_avg;
            if (metrics_flag)
            {
                std::ofstream outfile(metrics_file, std::ios::app);
                outfile << "COO,"
                        << cuda_op_map[cuda_op] << ","
                        << cuda_fill_map[cuda_fillmode] << ","
                        << cuda_diag_map[cuda_diagtype] << ","
                        << cuda_spsv_alg_map[cuda_alg] << ","
                        << cuda_datatype_map[cuda_datatype] << ","
                        << filename << ","
                        << cuda_time_bufferSize << ","
                        << cuda_time_analysis << ","
                        << cuda_time_solve_avg << ","
                        << cuda_time_total << "\n";
                outfile.close();
            }
            printf("%.6lf,", cuda_time_total);
            free(cuda_y);
        }
        else
        {
            ict_y = (DATA_TYPE *)alpha_malloc(m * sizeof(DATA_TYPE));
            alpha_fill_random(ict_y, 1, m);
            alpha_spsv();
            alpha_time_total = alpha_time_bufferSize + alpha_time_analysis + alpha_time_solve_avg;
            if (metrics_flag)
            {
                std::ofstream outfile(metrics_file, std::ios::app);
                outfile << "COO,"
                        << alpha_op_map[alpha_op] << ","
                        << alpha_fill_map[alpha_fillmode] << ","
                        << alpha_diag_map[alpha_diagtype] << ","
                        << alpha_spsv_alg_map[alpha_alg] << ","
                        << alpha_datatype_map[alpha_datatype] << ","
                        << filename << ","
                        << alpha_time_bufferSize << ","
                        << alpha_time_analysis << ","
                        << alpha_time_solve_avg << ","
                        << alpha_time_total << "\n";
                outfile.close();
            }
            printf("%.6lf,", alpha_time_bufferSize);
            printf("%.6lf,", alpha_time_analysis);
            printf("%.6lf,", alpha_time_solve_avg);
            printf("%.6lf,", alpha_time_total);

            // printf("\n");
            free(ict_y);
        }
    }
    else if (execute_type == 1)
    {
        cuda_y = (DATA_TYPE *)alpha_malloc(m * sizeof(DATA_TYPE));
        alpha_fill_random(cuda_y, 1, m);
        cuda_spsv();
        cuda_time_total = cuda_time_bufferSize + cuda_time_analysis + cuda_time_solve_avg;
        ict_y = (DATA_TYPE *)alpha_malloc(m * sizeof(DATA_TYPE));
        alpha_fill_random(ict_y, 1, m);
        alpha_spsv();
        alpha_time_total = alpha_time_bufferSize + alpha_time_analysis + alpha_time_solve_avg;
        double speedup_analysis = cuda_time_analysis / alpha_time_analysis;
        double speedup_solve = cuda_time_solve_avg / alpha_time_solve_avg;
        speedup = cuda_time_total / alpha_time_total;
        if (check_flag)
        {
            check((DATA_TYPE *)cuda_y, m, (DATA_TYPE *)ict_y, m, &error);
        }
        if (metrics_flag)
        {
            std::ofstream outfile(metrics_file, std::ios::app);
            outfile << "COO,"
                    << cuda_op_map[cuda_op] << ","
                    << cuda_fill_map[cuda_fillmode] << ","
                    << cuda_diag_map[cuda_diagtype] << ","
                    << cuda_spsv_alg_map[cuda_alg] << ","
                    << cuda_datatype_map[cuda_datatype] << ","
                    << filename << ","
                    << cuda_time_bufferSize << ","
                    << cuda_time_analysis << ","
                    << cuda_time_solve_avg << ","
                    << cuda_time_total << "\n";
            outfile << "COO,"
                    << alpha_op_map[alpha_op] << ","
                    << alpha_fill_map[alpha_fillmode] << ","
                    << alpha_diag_map[alpha_diagtype] << ","
                    << alpha_spsv_alg_map[alpha_alg] << ","
                    << alpha_datatype_map[alpha_datatype] << ","
                    << filename << ","
                    << alpha_time_bufferSize << ","
                    << alpha_time_analysis << ","
                    << alpha_time_solve_avg << ","
                    << alpha_time_total << ","
                    << speedup_analysis << ","
                    << speedup_solve << ","
                    << speedup;
            if (check_flag)
            {
                outfile << "," << error;
            }
            outfile << "\n";
            outfile.close();
        }
        printf("%.6lf,", speedup_analysis);
        printf("%.6lf,", speedup_solve);
        printf("%.6lf,", speedup);
        printf("\n");
        if (check_flag)
        {
            std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
            std::cout.precision(12);
            std::cout << "\nError: " << error << std::endl;
            std::cout << "\t i\t\tcuda_y, \t\tict_y\n";

            for (int i = 0, cnt = 0; i < m && cnt < 20; i++)
            {
                if (fabs(cuda_y[i] - ict_y[i]) > 1e-6)
                {
                    std::cout << cnt << ", \t";
                    std::cout << i << ", \t";
                    std::cout << cuda_y[i] << ", \t";
                    std::cout << ict_y[i] << std::endl;
                    cnt++;
                }
            }
            std::cout << std::endl;
        }
        free(ict_y);
        free(cuda_y);
    }
    free(x_val);

    return 0;
}
