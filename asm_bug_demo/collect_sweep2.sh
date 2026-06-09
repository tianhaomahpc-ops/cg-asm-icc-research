#!/usr/bin/env bash
# Sys3 wide sweeps:
#   EXP A: BASIC(scheme0)+ICC+CG, O=0..5 x L=0..6  -> find ICC level where overlap trend flips
#   EXP B: sASM(scheme3)+CG,      O=0..5 x L=0..5
# nx=48, 4 ranks, fix_level=1.  Run with: bash collect_sweep2.sh
set -u
NX=48; NP=4; FL=1
OUT=sweep2_nx${NX}_n${NP}.csv
echo "exp,scheme,O,L,iter,reason,err" > "$OUT"
K="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+"; }
getr(){ grep -oE "due to [A-Z_]+ iterations" | head -1 | sed -E 's/due to ([A-Z_]+).*/\1/'; }
gete(){ grep -oE "\|\|u-u\*\|\|/\|\|u\*\|\|=[0-9.e+-]+" | head -1 | sed 's/.*=//'; }

run(){ # exp scheme O L
  local e="$1" S="$2" O="$3" L="$4" out it rr er
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX -ksp_type cg $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  it=$(echo "$out"|geti); rr=$(echo "$out"|getr); er=$(echo "$out"|gete)
  printf "%s S%s O=%s L=%s | iter=%-5s %-14s err=%s\n" "$e" "$S" "$O" "$L" "${it:-NA}" "${rr:-NA}" "${er:-NA}"
  echo "$e,$S,$O,$L,${it:-NA},${rr:-NA},${er:-NA}" >> "$OUT"
}

echo "##### EXP A: BASIC(scheme0)+ICC, O=0..5 x L=0..6 #####"
for L in 0 1 2 3 4 5 6; do for O in 0 1 2 3 4 5; do run A 0 $O $L; done; done
echo "##### EXP B: sASM(scheme3), O=0..5 x L=0..5 #####"
for L in 0 1 2 3 4 5; do for O in 0 1 2 3 4 5; do run B 3 $O $L; done; done
echo "SWEEP2_DONE -> $OUT"
