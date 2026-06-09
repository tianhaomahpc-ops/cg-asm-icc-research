#!/usr/bin/env bash
set -u
NX=48; NP=4; FL=1
K="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
runi(){ local O="$1" L="$2" out it tt   # sASM + ICC(L)
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 3 -nx $NX -ksp_type cg $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  it=$(echo "$out"|geti); tt=$(echo "$out"|gett)
  printf "sASM+ICC   O=%-2s L=%-2s | iter=%-5s time=%ss\n" "$O" "$L" "${it:-NA}" "${tt:-NA}"; }
runc(){ local O="$1" out it tt        # BASIC + exact Cholesky (reference floor)
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX -ksp_type cg $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type cholesky 2>&1)
  it=$(echo "$out"|geti); tt=$(echo "$out"|gett)
  printf "BASIC+Chol O=%-2s      | iter=%-5s time=%ss\n" "$O" "${it:-NA}" "${tt:-NA}"; }
echo "### sASM+ICC push L at O=5 (O saturated) ###"
for L in 14 16 18 20; do runi 5 $L; done
echo "### BASIC + exact Cholesky reference (overlap saturated) ###"
for O in 2 3 5 8; do runc $O; done
echo "SWEEP4_DONE"
