add_definitions(-D__HIP__)
add_definitions(-D__HIPNEED__)   

add_subdirectory(kernel)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_util
    util/auxiliary.hip util/handle.hip util/simplelist.hip util/malloc.hip
)

if(NOT (ALPHA_BUILD_HYGON OR ALPHA_BUILD_PLAIN))
    list(APPEND alphasparse_util "util/thread.hip")
endif()

add_library(alphasparse_hip OBJECT ${alphasparse_util} ${alphasparse_source})

target_compile_definitions(alphasparse_hip PUBLIC __HIP_PLATFORM_AMD__)
target_compile_definitions(alphasparse_hip PUBLIC CUDA_ARCH=${CUDA_ARCH})
set_property(TARGET alphasparse_hip PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
target_link_libraries(alphasparse_hip PUBLIC
    hip::host
    hip::device
    roc::rocprim
    roc::rccl
    PRIVATE stdc++
)

target_include_directories(alphasparse_hip PRIVATE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>

    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/test/include>
)

add_subdirectory(test)
