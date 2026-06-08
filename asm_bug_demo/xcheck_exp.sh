#!/usr/bin/env bash
# Bit-identity cross-check for the new weighted-ASM schemes (5 and 6): MFEM vs PETSc.
set -u
NX=48; NP=4; FL=1
KSPBASE="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason"
geti() { grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+"; }

# seed dump
mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX $KSPBASE -ksp_type cg \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 0 -sub_ksp_type preonly \
  -sub_pc_type icc -sub_pc_factor_levels 0 >/dev/null 2>&1

xc() { # scheme ksp O L
  local S="$1" KT="$2" O="$3" L="$4" m p
  local C="-ksp_type $KT $KSPBASE -pc_type asm -pc_asm_type basic -pc_asm_overlap $O -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L"
  m=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX $C 2>&1 | geti)
  p=$(mpirun -n $NP ./pure_petsc_load -nx $NX -scheme $S $C 2>&1 | geti)
  printf "scheme=%s ksp=%-5s O=%s L=%s | MFEM=%-5s PETSc=%-5s %s\n" "$S" "$KT" "$O" "$L" "${m:-NA}" "${p:-NA}" "$([ "$m" = "$p" ] && echo EQ || echo DIFF)"
}

echo "## scheme 5 (D^-1 BASIC D^-1, CG) ##"
for O in 0 1 2; do for L in 0 1 2; do xc 5 cg $O $L; done; done
echo "## scheme 6 (D^-1 BASIC, GMRES) ##"
for O in 0 1 2; do xc 6 gmres $O 0; done
echo "## scheme 6 (D^-1 BASIC, CG) ##"
for O in 0 1 2; do xc 6 cg $O 0; done
echo "XCHECK_DONE"
