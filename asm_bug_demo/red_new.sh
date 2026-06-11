#!/usr/bin/env bash
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
red(){ grep -iE "MPI Reductions:" | head -1 | grep -oE "[0-9]\.[0-9]+e\+[0-9]+"; }
iti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
o=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme 8 -nx 48 $K -pc_type asm -pc_asm_type basic -pc_asm_overlap 3 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 2 -log_view 2>&1)
echo "sMRAS O3L2      : iter=$(echo "$o"|iti) reductions=$(echo "$o"|red)"
o=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 -pugrade 0.8 -localcheby 2 -nx 48 $K -pc_type asm -pc_asm_type basic -pc_asm_overlap 3 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 1 -log_view 2>&1)
echo "g0.8-d2 O3L1    : iter=$(echo "$o"|iti) reductions=$(echo "$o"|red)"
echo "RED_DONE"
