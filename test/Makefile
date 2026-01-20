.PHONY : test
SRC_DIR = .
# TEST_COMMON_SRCS = $(wildcard $(SRC_DIR)/*.c)
# TEST_COMMON_OBJS = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(TEST_COMMON_SRCS))
export TEST_COMMON_OBJS
ifeq ($(HIP_ON),1)
SUBDIRS += dcu
endif
ifeq ($(ARM_ON),1)
SUBDIRS += arm
endif
ifeq ($(HYGON_ON),1)
SUBDIRS += x86_64
endif

ifeq ($(PLAIN_ON),1)
ifeq ($(HAS_MKL),1)
SUBDIRS += plain
endif
endif
test :
	@for d in $(SUBDIRS) ; \
	do if test -d $$d; then \
	  $(MAKE) -C $$d $(@F) || exit 1 ; \
	fi; \
	done
# CFLAGS += $(INC) -I$(ROOT)/test/include $(DEFINE)
# CPPFLAGS += $(INC) -I$(ROOT)/test/include $(DEFINE)
# $(TEST_COMMON_OBJS) : $(OBJ_DIR)/%.o : %.c
# 	$(CC) -c -o $@ $^ $(CFLAGS)
