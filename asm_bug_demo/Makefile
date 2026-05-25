# Build asm_demo using the same MFEM/PETSc/HYPRE/MPI installation that
# the cardioid project links against (taken straight from MFEM's config.mk).
#
# Usage:
#   make            -> build ./asm_demo
#   make clean      -> remove object files / binary
#
MFEM_DIR := /Users/tianhaoma/spack/opt/spack/darwin-m1/mfem-4.9.0-dhws2hkr7xf7usvafygpgxn7zmgua645
CONFIG_MK := $(MFEM_DIR)/share/mfem/config.mk

include $(CONFIG_MK)

CXX      := $(MFEM_CXX)
CXXFLAGS := $(MFEM_FLAGS)
LDFLAGS  := $(MFEM_LIBS)

TARGET   := asm_demo
SRC      := asm_demo.cpp

# Pure-PETSc demo: no MFEM, no Hypre needed.  Pull just the PETSc
# include and library paths out of MFEM's MFEM_EXT_LIBS.
PETSC_DIR_DERIVED := /Users/tianhaoma/spack/opt/spack/darwin-m1/petsc-3.24.4-p7obww25tn67gjoykjws5bjdvfwbfpev
MPICC             := /Users/tianhaoma/spack/opt/spack/darwin-m1/openmpi-5.0.9-oloqi33z4barsfmcrz3c2drcnletrv2p/bin/mpicc

PURE_TARGET := pure_petsc_demo
PURE_SRC    := pure_petsc_demo.c
PURE_FEM_TGT := pure_petsc_fem
PURE_FEM_SRC := pure_petsc_fem.c
PURE_LOAD_TGT := pure_petsc_load
PURE_LOAD_SRC := pure_petsc_load.c
PURE_INC    := -I$(PETSC_DIR_DERIVED)/include
PURE_LIB    := -Wl,-rpath,$(PETSC_DIR_DERIVED)/lib -L$(PETSC_DIR_DERIVED)/lib -lpetsc

all: $(TARGET) $(PURE_TARGET) $(PURE_FEM_TGT) $(PURE_LOAD_TGT)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@ $(LDFLAGS)

$(PURE_TARGET): $(PURE_SRC)
	$(MPICC) -O3 -std=c11 $(PURE_INC) $(PURE_SRC) -o $@ $(PURE_LIB)

$(PURE_FEM_TGT): $(PURE_FEM_SRC)
	$(MPICC) -O3 -std=c11 $(PURE_INC) $(PURE_FEM_SRC) -o $@ $(PURE_LIB)

$(PURE_LOAD_TGT): $(PURE_LOAD_SRC)
	$(MPICC) -O3 -std=c11 $(PURE_INC) $(PURE_LOAD_SRC) -o $@ $(PURE_LIB)

clean:
	$(RM) $(TARGET) $(PURE_TARGET) $(PURE_FEM_TGT) $(PURE_LOAD_TGT) *.o

.PHONY: all clean
