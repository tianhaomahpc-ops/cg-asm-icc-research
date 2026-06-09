#!/usr/bin/env bash
# sASM (scheme 3) + CG: push O and L higher to find the iteration-count floor. Sys3, nx=48, 4 ranks.
set -u
NX=48; NP=4; FL=1
OUT=sweep3_nx${NX}_n${NP}.csv
echo "scheme,O,L,iter,reason,time" > "$OUT"
K="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+"; }
getr(){ grep -oE "due to [A-Z_]+ iterations" | head -1 | sed -E 's/due to ([A-Z_]+).*/\1/'; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
run(){ local O="$1" L="$2" out it rr tt
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 3 -nx $NX -ksp_type cg $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  it=$(echo "$out"|geti); rr=$(echo "$out"|getr); tt=$(echo "$out"|gett)
  printf "sASM O=%-2s L=%-2s | iter=%-5s %-14s time=%ss\n" "$O" "$L" "${it:-NA}" "${rr:-NA}" "${tt:-NA}"
  echo "3,$O,$L,${it:-NA},${rr:-NA},${tt:-NA}" >> "$OUT"
}
echo "### extend L at O=5 (O nearly saturated) ###"
for L in 6 7 8 9 10 12; do run 5 $L; done
echo "### confirm O floor at L=8 ###"
for O in 6 7 8 10; do run $O 8; done
echo "### far corner (high O x high L) ###"
run 8 10; run 8 12; run 10 10; run 10 12
echo "SWEEP3_DONE -> $OUT"
