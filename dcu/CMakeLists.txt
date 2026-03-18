add_definitions(-D__DCU__)

add_subdirectory(kernel)
add_subdirectory(format)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    test/test_common.cpp test/args.cpp test/check.cpp 
    test/io.cpp test/norm.cpp
)

set(alphasparse_util
    util/malloc_dcu.cpp util/handle.cpp
)

set(alphasparse_cpu_util
    ${PROJECT_SOURCE_DIR}/hygon/util/malloc.cpp 
)

set(alphasparse_cpu_format
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_create_coo.cpp  
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_coo.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_convert_csr.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_destroy.cpp
    ${PROJECT_SOURCE_DIR}/hygon/format/alphasparse_transpose.cpp
    ${PROJECT_SOURCE_DIR}/hygon/util/thread.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/util/partition.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/util/prefix_sum.cpp 
    ${PROJECT_SOURCE_DIR}/hygon/util/bisearch.cpp
)

add_library(alphasparse ${alphasparse_headers_public} ${alphasparse_test} ${alphasparse_util} ${alphasparse_cpu_util} ${alphasparse_format} ${alphasparse_cpu_format} ${alphasparse_source} )

# target_link_libraries(alphasparse ${HIP_LIBRARIES})

target_link_libraries(alphasparse PUBLIC
    roc::hipsparse
    hip::host
    hip::device
    roc::rocprim
    roc::rccl
    # hip::amdhip64
)

target_include_directories(alphasparse PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

add_subdirectory(test)
