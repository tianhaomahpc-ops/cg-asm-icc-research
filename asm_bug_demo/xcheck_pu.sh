#!/usr/bin/env bash
# Bit-identity cross-check for scheme 7: MFEM (asm_demo) vs pure PETSc (pure_petsc_load).
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+"; }
# seed dump (default Sys3)
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 0 -nx 48 $K -pc_type asm -pc_asm_type basic -pc_asm_overlap 0 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0 >/dev/null 2>&1
for EPS in 0.0 0.7 1.0; do for O in 1 2 3; do for L in 0 2; do
  C="-pc_type asm -pc_asm_type basic -pc_asm_overlap $O -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L"
  m=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 -pueps $EPS -nx 48 $K $C 2>&1 | geti)
  p=$(mpirun -n 4 ./pure_petsc_load -nx 48 -scheme 7 -pueps $EPS $K $C 2>&1 | geti)
  printf "eps=%-4s O=%s L=%s | MFEM=%-5s PETSc=%-5s %s\n" "$EPS" "$O" "$L" "${m:-NA}" "${p:-NA}" "$([ "$m" = "$p" ] && echo EQ || echo DIFF)"
done; done; done
echo "XPU_DONE"
