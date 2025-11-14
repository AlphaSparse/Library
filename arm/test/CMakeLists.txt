function(add_alphasparse_example TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  include_directories(./include)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})
  # add_compile_definitions(__MKL__)
  # MESSAGE(STATUS "TEST_TARGET = ${TEST_TARGET}.${TEST_SOURCE}")
  # target_compile_definitions(${TEST_TARGET} PUBLIC CUDA_ARCH=${CUDA_ARCH})
  # set_property(TARGET ${TEST_TARGET} PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})

  # Include common client headers
  target_include_directories(${TEST_TARGET} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  # Linker dependencies
  target_link_libraries(${TEST_TARGET} PUBLIC
      alphasparse      
      m
      dl
      )
  # Target properties
#   set_target_properties(${TEST_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endfunction()

add_alphasparse_example(level2/mv_hygon_test.cpp)
add_alphasparse_example(level2/sv_hygon_test.cpp)
add_alphasparse_example(level2/sv_csr_s_hygon_test.cpp)
add_alphasparse_example(level3/mm_hygon_test.cpp)
add_alphasparse_example(level3/spmm_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csr_d_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csr_c_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csr_z_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csc_s_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csc_d_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csc_c_hygon_test.cpp)
add_alphasparse_example(level3/spmm_csc_z_hygon_test.cpp)
add_alphasparse_example(level3/spmmd_hygon_test.cpp)
add_alphasparse_example(level3/trsm_hygon_test.cpp)
add_alphasparse_example(level3/trsm_csr_s_hygon_test.cpp)
add_alphasparse_example(level3/trsm_d_hygon_test.cpp)
add_alphasparse_example(level3/trsm_c_hygon_test.cpp)
add_alphasparse_example(level3/add_s_csr_x86_64_test.cpp)
add_alphasparse_example(level3/set_s_csr_x86_64_test.cpp)
