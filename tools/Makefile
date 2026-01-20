
SRC_DIR = .
vpath %.c $(SRC_DIR)

TOOL_SRC = $(wildcard $(SRC_DIR)/*.c)
OBJ = $(patsubst %.c, $(OBJ_DIR)/%.o, $(TOOL_SRC) )
TOOL_BIN = $(patsubst %.c, $(BIN_DIR)/%, $(TOOL_SRC) )

CFLAGS = -w -g -O2 -std=c11 -lm

include $(ROOT)/Makefile.tail
