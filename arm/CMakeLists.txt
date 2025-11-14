add_definitions(-D__ARM__)   

add_subdirectory(kernel)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    test/test_common.cpp test/args.cpp test/check.cpp 
    test/io.cpp test/norm.cpp
)

file(GLOB FORMAT_SOURCES "../hygon/format/*.cpp")
file(GLOB FORMAT_HEADERS "../hygon/format/*.hpp")
file(GLOB UTIL_SOURCES "../hygon/util/*.cpp")

set(alphasparse_cpu_common ${FORMAT_SOURCES} ${FORMAT_HEADERS} ${UTIL_SOURCES})

add_library(alphasparse_arm ${alphasparse_headers_public} ${alphasparse_test} ${alphasparse_cpu_common} ${ASM_SOURCES} ${alphasparse_source} )

target_include_directories(alphasparse_arm PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                        PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/hygon>)

add_subdirectory(test)
