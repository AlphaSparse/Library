cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project("AlphaSparse-Library" LANGUAGES CXX C)

# --- 基本编译设置 ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -gdwarf-4")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -gdwarf-4")
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -gdwarf-4")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- 构建选项定义 ---
option(ALPHA_BUILD_CUDA "Compile kernels for NVIDIA GPUs" OFF)
option(ALPHA_BUILD_HIP "Compile kernels for AMD or NVIDIA GPUs" ON)
option(ALPHA_BUILD_HYGON "Compile kernels for Hygon CPUs" OFF)
option(ALPHA_BUILD_ARM "Compile kernels for ARM CPUs" OFF)
option(ALPHA_BUILD_PLAIN "Compile serial kernels for x86 CPUs (Reference Implementation)" OFF)


# ====================================================================
# 【核心修改】条件化MKL依赖
# 只有在构建 PLAIN 或 HYGON 版本时，才检查并引入MKL依赖
# ====================================================================
set(MKL_IS_REQUIRED OFF)
if(ALPHA_BUILD_PLAIN OR ALPHA_BUILD_HYGON)
    set(MKL_IS_REQUIRED ON)
endif()

if(MKL_IS_REQUIRED)
    message(STATUS "MKL support is required for the current build configuration (PLAIN or HYGON).")
    # 检查MKLROOT环境变量
    if(NOT DEFINED ENV{MKLROOT})
        message(FATAL_ERROR "MKLROOT environment variable is not set, but is required for PLAIN/HYGON builds. Please source the oneAPI setvars.sh script before running cmake.")
    endif()

    message(STATUS "Using MKL from MKLROOT: $ENV{MKLROOT}")

    # 添加MKL的头文件和库路径
    include_directories($ENV{MKLROOT}/include)
    link_directories($ENV{MKLROOT}/lib/intel64)

    # 统一添加__MKL__宏定义
    add_definitions(-D__MKL__)
else()
    message(STATUS "MKL support is NOT required for the current build configuration.")
endif()
# ======================= 修改结束 ============================


# --- 各后端构建逻辑 ---

if(ALPHA_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit)
    set (CUDA_ARCH 70)
    add_subdirectory(cuda)
    string(APPEND CMAKE_CUDA_FLAGS " -gencode=arch=compute_70,code=sm_70")
    string(APPEND CMAKE_CUDA_FLAGS "  -Xcompiler -Wall -fPIC -D_FORCE_INLINES --expt-extended-lambda -use_fast_math --expt-relaxed-constexpr")
    string(APPEND CMAKE_CUDA_FLAGS " -I/home/guochengxin/cusplibrary-cuda10")
endif()

if(ALPHA_BUILD_HIP)
    set(CMAKE_HIP_HOST_COMPILER /opt/dtk/llvm/bin/clang++)
    set(CMAKE_HIP_COMPILER /opt/dtk/llvm/bin/clang++)
    enable_language(HIP)
    set (HIP_ARCH 70)
    set(HIP_CLANG_INCLUDE_PATH  /opt/dtk/llvm/lib/clang/15.0.0/lib)
    find_package(hip REQUIRED)
    find_package(hipsparse REQUIRED)
    find_package(rocprim REQUIRED)
    find_package(rccl REQUIRED)
    find_package(rocsparse REQUIRED)
    add_subdirectory(hip)
endif()

if(ALPHA_BUILD_DCU)
    enable_language(HIP)
    find_package(HIP REQUIRED)
    include_directories(${HIP_INCLUDE_DIRS})
    set (HIP_ARCH 70)
    set(CMAKE_CXX_COMPILER hipcc)
    set(__HIP_PLATFORM_AMD__ 1)
    find_package(hipsparse REQUIRED)
    find_package(rocprim REQUIRED)
    find_package(rccl REQUIRED)
    find_package(rocsparse REQUIRED)
    add_subdirectory(dcu)
endif()

if(ALPHA_BUILD_HYGON)
    # 【已修改】不再需要在这里添加 -D__MKL__，已移至顶部条件块
    enable_language(C ASM)
    find_package(OpenMP REQUIRED)
    Set(CMAKE_C_FLAGS "$CMAKE_C_FLAGS} -fopenmp")
    Set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    Set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(hygon)
endif()

if(ALPHA_BUILD_ARM)
    enable_language(C ASM)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(arm)
endif()

if(ALPHA_BUILD_PLAIN)
    # 【已修改】不再需要在这里添加 -D__MKL__，已移至顶部条件块
    add_definitions(-D__PLAIN__)
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(plain)
endif()


# --- 目标库的构建和链接 ---
set(OBJECT_LIBS "")

if(ALPHA_BUILD_CUDA)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_cuda>)
endif()

if(ALPHA_BUILD_HYGON)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_hygon>)
endif()

if(ALPHA_BUILD_ARM)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_arm>)
endif()

if(ALPHA_BUILD_PLAIN)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_ref>)
endif()

if(ALPHA_BUILD_HIP)
    list(APPEND OBJECT_LIBS $<TARGET_OBJECTS:alphasparse_hip>)
endif()

if(OBJECT_LIBS)
    add_library(alphasparse SHARED ${OBJECT_LIBS})
    if(ALPHA_BUILD_HIP)
        target_link_libraries(alphasparse PUBLIC
            hip::host
            roc::hipsparse
            roc::rocprim
            roc::rccl
        )
    endif()
    target_include_directories(alphasparse PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    )
else()
    message(WARNING "No source directories are selected for building. No library will be created.")
endif()