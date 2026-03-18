add_definitions(-D__CUDA__)

add_subdirectory(kernel)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    test/args.cu test/check.cu test/io.cu test/test_common.h
    test/check_r.cu
)

set(alphasparse_util
    util/auxiliary.cu util/handle.cu util/malloc.cu util/simplelist.cu util/thread.cu
)

add_library(alphasparse_cuda ${alphasparse_headers_public} ${alphasparse_test} ${alphasparse_util} ${alphasparse_source})
target_compile_definitions(alphasparse_cuda PUBLIC __CUDA_NO_HALF2_OPERATORS__ __CUDA_NO_BFLOAT162_OPERATORS__)
target_compile_definitions(alphasparse_cuda PUBLIC CUDA_ARCH=${CUDA_ARCH})
set_property(TARGET alphasparse_cuda PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(alphasparse_cuda PUBLIC
    CUDA::cudart
    CUDA::cudart_static
    CUDA::cusparse
    CUDA::cusparse_static
    CUDA::cublas
    CUDA::cublas_static
)
target_include_directories(alphasparse_cuda PRIVATE ${CUDAToolkit_INCLUDE_DIRS} $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

add_subdirectory(test)
