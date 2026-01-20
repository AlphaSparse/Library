CFLAGS += $(INC) $(DEFINE)
CPPFLAGS += $(INC) $(DEFINE)

SUBDIRS = core kernel util

ifeq ($(PLAIN_ON),1)
SUBDIRS += plain
endif
ifeq ($(HIP_ON),1)
SUBDIRS += util_cpp
# SUBDIRS += tuning
endif
ifeq ($(ARM_ON),1)
# SUBDIRS += tuning
endif

.PHONY : lib

lib :
	@for d in $(SUBDIRS) ; \
	do if test -d $$d; then \
	  $(MAKE) -C $$d $(@F) || exit 1 ; \
	fi; \
	done
