#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
#MPI_ROOT  = /usr/lib/x86_64-linux-gnu/openmpi
#LAUNCHER  = /usr/bin/mpirun -n 2
MPI_ROOT  = $(HOME)/ompi_for_gpu/ompi
LAUNCHER  = $(HOME)/ompi_for_gpu/ompi/bin/mpirun -n 2

#===============================================================================
# Program name & source code list
#===============================================================================

mpi-program = main-mpi
nccl-program = main-nccl

mpi-source = main-mpi.cu
nccl-source = main-nccl.cu

mpi-obj = $(mpi-source:.cu=.o)
nccl-obj = $(nccl-source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 -I$(MPI_ROOT)/include -Wall

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -lrccl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(mpi-program): $(mpi-obj) Makefile
	$(CC) $(CFLAGS) $(mpi-obj) -o $@ $(LDFLAGS)

$(nccl-program): $(nccl-obj) Makefile
	$(CC) $(CFLAGS) $(nccl-obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(mpi-program) $(mpi-obj)
	rm -rf $(nccl-program) $(nccl-obj)

run: $(mpi-program) $(nccl-program)
	$(LAUNCHER) ./$(mpi-program)
	$(LAUNCHER) ./$(nccl-program)
