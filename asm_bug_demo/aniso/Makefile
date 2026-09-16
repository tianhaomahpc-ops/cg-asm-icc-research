# Self-contained sigma-tensor harness.  No MFEM, no PETSc, no MPI -- just a C
# IC(0) kernel + numpy/scipy, so it runs anywhere (including CI / a laptop with
# no Spack stack).  It is the REFERENCE implementation for the two new
# partition-of-unity families; the MFEM/PETSc versions live one directory up.
#
#   make            -> libic0.so
#   make sanity     -> the three correctness anchors
#   make figs       -> all sweeps + the three PNGs
CC      ?= gcc
CFLAGS  ?= -O3 -fPIC -shared
PY      ?= python3

all: libic0.so

libic0.so: ic0kernel.c
	$(CC) $(CFLAGS) $< -o $@ -lm

sanity: libic0.so
	$(PY) sanity.py

figs: libic0.so
	$(PY) fig1_contrast.py
	$(PY) fig2_pou.py
	$(PY) fig3_coarse.py
	$(PY) fig4_scale.py
	$(PY) fig5_enrich.py
	$(PY) plot_all.py

clean:
	$(RM) libic0.so *.csv *.png
	$(RM) -r __pycache__

.PHONY: all sanity figs clean
