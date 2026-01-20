SRC_DIR = $(shell find . -type d)

vpath %.cpp $(SRC_DIR)

TEST_SRC = $(foreach d,$(SRC_DIR), $(wildcard $(d)/*.cpp) )
# OBJ = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_SRC) )
OBJCPP = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(TEST_SRC)) )
TEST_BIN = $(patsubst %.cpp, $(BIN_DIR)/%, $(notdir $(TEST_SRC)) )

CFLAGS += $(INC) $(LIB) $(DEFINE) -l:libalpha_spblas.a -lm -I$(ROOT)/test

CPPFLAGS += $(INC) $(DEFINE) -I$(ROOT)/test
LDFLAGS += -l:libalpha_spblas.a -lm -lstdc++
CFLAGS += $(INC) $(DEFINE) -I$(ROOT)/test
include $(ROOT)/Makefile.tail
