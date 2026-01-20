# 是否开启openmp，1 开启，0 不开启 单线程版本。
OPENMP = 1
# 是否生成汇编代码，生成的汇编代码在 asm 目录下。1 生成，0 不生成。
ASM_COMPILE = 0
# MKL_INT 和 ALPHA_INT 是否使用64位，1 使用64位，0 使用32位。
INT_64 = 0
# 表示编不编译
HIP_ON = $(shell echo $${HIP_ON:-0})
# PLAIN 依赖 mkl
PLAIN_ON = $(shell echo $${PLAIN_ON:-1})
# DEBUG
DEBUG_ON = $(shell echo $${DEBUG_ON:-0})
ARM_ON = $(shell echo $${ARM_ON:-0})
HYGON_ON = $(shell echo $${HYGON_ON:-0})
HAS_MKL = $(shell echo $${HAS_MKL:-0})
HYGON_ON = 0
HAS_MKL = 0
PLAIN_ON = 0
ROOT = $(shell pwd)
OS := $(shell uname -s)

LIB_DIR = $(ROOT)/lib
INC_DIR = $(ROOT)/include
OBJ_DIR = $(ROOT)/obj
BIN_DIR = $(ROOT)/bin
ASM_DIR = $(ROOT)/asm

ifeq ($(DEBUG_ON), 1)
CPPFLAGS += -ggdb
endif

ifeq ($(HIP_ON), 1)
ROCM_DIR = /public/software/compiler/rocm/rocm-3.9.1
ROCSP_DIR = /home/gcx/csparse
endif

LIBNAME = libalpha_spblas

INC += -I$(INC_DIR)

ifeq ($(HIP_ON), 1)
INC += -I$(ROCM_DIR)/hip/include
endif

export ROOT LIB_DIR INC_DIR OBJ_DIR BIN_DIR ASM_DIR INC DEFINE LIBNAME OPENMP ASM_COMPILE INT_64 HIP_ON PLAIN_ON HAS_MKL ARM_ON HYGON_ON
GCC_VERSION_GE9_3_1 := $(shell expr `gcc --version | awk -F" " '/^gcc/{print $$3}' |  tr -d '.' ` \>= 931)

ifneq ($(OS), Darwin)
CPUVENDOR := $(shell lscpu | awk -F"[ ;]" '/^Vendor/ {print $$NF}' )
endif

MAKE = make
CC = gcc
HCC= hipcc
AR = ar
LDFLAGS = -L$(LIB_DIR)

CPPFLAGS += -std=c++11 -fPIC -Ofast -I$(INC_DIR)
CFLAGS += -Ofast
CFLAGS += -std=c11
CFLAGS += -g
ifeq ($(OPENMP), 1)
CFLAGS += -fopenmp
CPPFLAGS += -fopenmp
endif
# CFLAGS += -mcmodel=large
CFLAGS += -Wall
# CFLAGS += -Wextra
# CFLAGS += -Werror
CFLAGS += -Wno-unused-result
CFLAGS += -Wno-unused-parameter
CFLAGS += -Wno-unused-function
CFLAGS += -Wno-unused-variable
CFLAGS += -fstack-protector-all
CFLAGS += -fPIC
# CFLAGS += -Wl,-z,relro -Wl,-z,noexecstack -Wl,-z,now

ARCH = $(shell uname -m)
ifeq ($(ARCH), aarch64)
CC = gcc
DEFINE += -D__aarch64__
ifeq ($(GCC_VERSION_GE9_3_1), 1)
CFLAGS += -march=armv8.3-a
DEFINE += -DCMLA
endif
endif
ifeq ($(ARCH), x86_64)
CC = gcc
ifneq ($(OS), Darwin)
ifeq ($(CPUVENDOR), GenuineIntel)
CC = icc
endif
endif
CFLAGS += -m64
CFLAGS += -march=native
DEFINE += -D__x86_64__
ifeq ($(HAS_MKL), 1)
DEFINE += -D__MKL__
endif
endif
ifeq ($(ARCH), ppc)
CC = gcc
CFLAGS += -mtune=native
CFLAGS += -arch ppc
DEFINE += -Dppc
endif
ifeq ($(ARCH), ppc64)
CC = gcc
CFLAGS += -mtune=native
CFLAGS += -arch ppc64
DEFINE += -Dppc64
endif
ifeq ($(CC), gcc)
endif
ifeq ($(CC), icc)
endif

ifeq ($(INT_64), 1)
DEFINE += -DMKL_ILP64
DEFINE += -DALPHA_INT=int64_t
endif

ifeq ($(HIP_ON), 1)
CEXTRAFLAGS += -lstdc++ -L$(ROCM_DIR)/hip/lib -L$(ROCM_DIR)/rocsparse/lib -lamdhip64 -lrocsparse
LDFLAGS += $(CEXTRAFLAGS)
DEFINE += -D__DCU__
endif

ifeq ($(PLAIN_ON), 1)
DEFINE += -D__PLAIN__
endif

export MAKE CC HCC AR CFLAGS CPPFLAGS CEXTRAFLAGS ARCH LDFLAGS

.PHONY : clean lib so test

all : lib so test

lib :
	$(MAKE) -C src $(@F)

# FIXME: this will not work on macOS:
so : lib
	$(CC) -shared -o $(LIB_DIR)/$(LIBNAME).so -Wl,--whole-archive $(LIB_DIR)/$(LIBNAME).a -Wl,--no-whole-archive

# TODO: implement tests for PPC.
test : lib
	$(MAKE) -C test $(@F)

# TODO: implement tool target (needs fixes to makefiles).
# tool :
# 	$(MAKE) -C tools $(@F)

clean :
	find $(OBJ_DIR) -type f ! -name "*xgb*.o" -delete
	rm -rf $(BIN_DIR)/*
	rm -rf $(LIB_DIR)/*
