set(ALPHA_HIP_TEST_COMMON_SOURCES
    args.hip
    io.hip
    check.hip
    check_r.hip
    warmup.hip
)

add_library(alphasparse_hip_test_utils_objs OBJECT 
    ${ALPHA_HIP_TEST_COMMON_SOURCES}
)

target_include_directories(alphasparse_hip_test_utils_objs PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

function(add_alphasparse_example TEST_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_SOURCE})

    target_compile_definitions(${TEST_TARGET} PUBLIC __HIP_PLATFORM_HCC__)
    target_compile_definitions(${TEST_TARGET} PUBLIC CUDA_ARCH=${CUDA_ARCH})
    set_property(TARGET ${TEST_TARGET} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

    target_include_directories(${TEST_TARGET} PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    )
    target_link_libraries(${TEST_TARGET} PUBLIC
        roc::hipsparse
        hip::host
        hip::device
        roc::rocprim
        roc::rccl
        alphasparse
        $<TARGET_OBJECTS:alphasparse_hip_test_utils_objs>
    )

    if(ALPHA_BUILD_PLAIN)
        find_package(OpenMP REQUIRED)
        target_link_libraries(${TEST_TARGET} PUBLIC
            mkl_cdft_core
            mkl_intel_lp64
            mkl_gnu_thread
            mkl_core
            pthread
            dl
            OpenMP::OpenMP_CXX
        )
    endif()
endfunction()

add_alphasparse_example(level3/spgemm_csr_r_f32_test.hip)
# add_alphasparse_example(level3/spgemm_csr_r_f64_test.hip)

add_alphasparse_example(level2/spmv_csr_r_f32_test.hip)
add_alphasparse_example(level2/spmv_csr_r_f64_test.hip)
# add_alphasparse_example(level2/spmv_csr_r_f64_test_metrics.hip)
# add_alphasparse_example(level2/spmv_csr_c_f32_test_metrics.hip)
# add_alphasparse_example(level2/spmv_csr_c_f64_test_metrics.hip)

add_alphasparse_example(level2/spsv_csr_r_f64_test_metrics.hip)
add_alphasparse_example(level2/spsv_csr_r_f32_test_metrics.hip)
add_alphasparse_example(level2/spsv_coo_r_f64_test_metrics.hip)
add_alphasparse_example(level2/spsv_coo_r_f32_test_metrics.hip)

add_alphasparse_example(level3/spmm_csr_row_r_f32_test_hip_metrics.hip)
add_alphasparse_example(level3/spmm_csr_row_r_f32_test_alpha_metrics.hip)
add_alphasparse_example(level3/spmm_csr_row_r_f64_test_hip_metrics.hip)
add_alphasparse_example(level3/spmm_csr_row_r_f64_test_alpha_metrics.hip)

