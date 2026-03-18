function(add_alphasparse_example TEST_SOURCE)
  get_filename_component(TEST_TARGET ${TEST_SOURCE} NAME_WE)
  include_directories(./include)
  include_directories(${PROJECT_SOURCE_DIR}/hygon/test/include)
  add_executable(${TEST_TARGET} ${TEST_SOURCE})

  # Include common client headers
  target_include_directories(${TEST_TARGET} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
  target_link_libraries(${TEST_TARGET} PUBLIC
        mkl_cdft_core 
        mkl_intel_lp64 
        mkl_gnu_thread
        mkl_core
        pthread
        dl      
        OpenMP::OpenMP_CXX
        alphasparse
      )
  # Linker dependencies
  if(ALPHA_BUILD_HIP)
    target_link_libraries(${TEST_TARGET} PUBLIC
        roc::hipsparse
      )  
  endif()
endfunction()

file(GLOB ALL_FILES "./*.cpp")
foreach(TEST_FILE ${ALL_FILES})
    message("Processing file: ${TEST_FILE}")
    add_alphasparse_example(${TEST_FILE})
endforeach()

