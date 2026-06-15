#!/usr/bin/env bash
# 3D Sys3 (1 Dirichlet + 5 Neumann Laplace, nx=48) CG residual histories via -ksp_monitor.
set -u
NX=48; NP=4; OUT=infoprop_out
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-8 -ksp_atol 1e-14 -ksp_max_it 4000 -ksp_monitor"
PC="-pc_type asm -pc_asm_type basic -pc_asm_overlap 2 -sub_ksp_type preonly"
grab(){ grep -oE "[0-9]+ KSP Residual norm [0-9.eE+-]+" | awk '{print $1, $5}'; }
mpirun -n $NP ./asm_demo -fix_level 1 -scheme 0 -nx $NX $K $PC -sub_pc_type icc -sub_pc_factor_levels 0 2>&1 | grab > $OUT/res3d_BASIC_icc.txt
mpirun -n $NP ./asm_demo -fix_level 1 -scheme 3 -nx $NX $K $PC -sub_pc_type icc -sub_pc_factor_levels 0 2>&1 | grab > $OUT/res3d_sASM_icc.txt
mpirun -n $NP ./asm_demo -fix_level 1 -scheme 0 -nx $NX $K $PC -sub_pc_type cholesky 2>&1 | grab > $OUT/res3d_BASIC_chol.txt
mpirun -n $NP ./asm_demo -fix_level 1 -scheme 3 -nx $NX $K $PC -sub_pc_type cholesky 2>&1 | grab > $OUT/res3d_sASM_chol.txt
echo RES3D_DONE; wc -l $OUT/res3d_*.txt
