add_subdirectory(kernel)

set(alphasparse_headers_public
    ${PROJECT_SOURCE_DIR}/include/alphasparse.h
)

set(alphasparse_test
    ../hygon/test/test_common.cpp ../hygon/test/args.cpp ../hygon/test/check.cpp 
    ../hygon/test/io.cpp ../hygon/test/norm.cpp
)

file(GLOB FORMAT_SOURCES "../hygon/format/*.cpp")
file(GLOB FORMAT_HEADERS "../hygon/format/*.hpp")
file(GLOB UTIL_SOURCES "../hygon/util/*.cpp")
if(ALPHA_BUILD_HYGON)
    set(alphasparse_cpu_common "")    
    message(WARNING "HYGON will be Included.")
else()
    set(alphasparse_cpu_common ${FORMAT_SOURCES} ${FORMAT_HEADERS} ${UTIL_SOURCES} ${alphasparse_test})
    message(WARNING "HYGON will NOT be Included and LINKING HYGON commons")
endif()
add_library(alphasparse_ref OBJECT ${alphasparse_headers_public} ${alphasparse_cpu_common} ${ASM_SOURCES} ${alphasparse_source} )

target_include_directories(alphasparse_ref PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                       PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/hygon>)

add_subdirectory(test)
