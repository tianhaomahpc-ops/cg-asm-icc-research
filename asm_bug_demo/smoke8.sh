#!/usr/bin/env bash
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
echo "== scheme 8 (sMRAS, theta=1) O=1 L=0 : full diagnostics =="
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 8 -nx 48 $K \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 \
  -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0 2>&1 \
  | grep -iE "sMRAS|RESULT|SOLN|converged|DIVERG|error|unsupported|No support" | head -6
echo "== scheme 7 graded q=0.8 + localcheby 2, O=2 L=1 (vs scheme4 48) =="
mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 -pugrade 0.8 -localcheby 2 -nx 48 $K \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
  -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 1 2>&1 \
  | grep -iE "RESULT|SOLN|converged|DIVERG" | head -3
echo "SMOKE8_DONE"
