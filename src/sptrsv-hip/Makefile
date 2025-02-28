#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o sptrsv_syncfree.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -Wall \
          -DWARP_SIZE=64 -DWARP_PER_BLOCK=8 \
          -DVALUE_TYPE=double -I../sptrsv-sycl

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../sptrsv-sycl/main.cpp \
	../sptrsv-sycl/mmio.h \
	../sptrsv-sycl/read_mtx.h \
	../sptrsv-sycl/sptrsv.h \
	../sptrsv-sycl/tranpose.h
	$(CC) $(CFLAGS) -c $< -o $@

sptrsv_syncfree.o : sptrsv_syncfree.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../sptrsv-sycl/lp1.mtx 2000
