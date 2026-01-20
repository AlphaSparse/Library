
SRC_DIR = .
vpath %.c $(SRC_DIR)

TEST_SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_SRC) )
TEST_BIN = $(patsubst $(SRC_DIR)/%.c, $(BIN_DIR)/%, $(TEST_SRC) )

CFLAGS += -lmkl_cdft_core -lmkl_intel_thread -lmkl_core -lpthread -lmkl_rt -ldl
ifeq ($(INT_64),1)
CFLAGS += -lmkl_intel_ilp64
else
CFLAGS += -lmkl_intel_lp64
endif
ifeq ($(OPENMP),1)
CFLAGS += -liomp5
endif

CFLAGS += $(INC) $(LIB) $(DEFINE) -l:libalpha_spblas.a -lm

include $(ROOT)/Makefile.tail
