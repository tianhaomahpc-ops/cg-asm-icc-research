#!/usr/bin/env bash
# collect_twolevel_fair.sh -- the same two-level comparison, but stopped on the
# TRUE residual instead of the preconditioned one.
#
# Why this file exists.  Comparing at fixed -ksp_rtol with
# -ksp_norm_type preconditioned is NOT a fair comparison across preconditioners:
# the test is on ||r||_M, and adding a coarse level changes M (lambda_max goes
# from ~1.17 to ~1.99), so the same ||r||_M corresponds to a LOOSER ||b-Ax||.
# Measured at nx=48 / 8 ranks / r=100: harm+coarse stopped at ||r||/||b|| =
# 3.1e-5 while harm and sASM stopped at 2.6e-6 -- a 10x slacker answer that
# flattered the coarse level by ~30 iterations.  -ksp_norm_type unpreconditioned
# holds every method to the same delivered accuracy.
#
# usage: ./collect_twolevel_fair.sh [nx] [icc_level] [maxranks]
set -u
NX=${1:-48}; L=${2:-0}; MAXP=${3:-16}
OUT="twolevel_fair_nx${NX}_L${L}.csv"
MPIRUN=${MPIRUN:-mpirun}
echo "ranks,ratio,method,O,iter,true_res,unorm" > "$OUT"
KSP="-ksp_type cg -ksp_norm_type unpreconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 6000"

LABELS=(  "BASIC" "sASM"  "harm"     "harm+coarse"        "mult+coarse" )
SCHEMES=( 0       3       7          7                    7             )
FLAGS=(   ""      ""      "-puharm"  "-puharm -pucoarse"  "-pucoarse"   )

for R in 1 100; do
for NP in 2 4 8 16 32; do
  [ "$NP" -gt "$MAXP" ] && continue
  for k in "${!LABELS[@]}"; do
    lab=${LABELS[$k]}; sc=${SCHEMES[$k]}; fl=${FLAGS[$k]}
    out=$($MPIRUN --allow-run-as-root --oversubscribe -n "$NP" ./asm_demo \
          -nx "$NX" -fix_level 1 -scheme "$sc" $fl -aniso "$R" -fiber 1,1,1 \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
          $KSP 2>&1)
    it=$( sed -n 's/.*iters=\([0-9]*\).*/\1/p'                <<<"$out"|head -1)
    tr_=$(sed -n 's@.*||r||/||b||=\([0-9.e+-]*\).*@\1@p'      <<<"$out"|head -1)
    un=$( sed -n 's@.*||u||=\([0-9.e+-]*\).*@\1@p'            <<<"$out"|head -1)
    echo "$NP,$R,$lab,2,${it:-NA},${tr_:-NA},${un:-NA}" >> "$OUT"
    printf "  r=%-4s P=%-3s %-12s -> iter=%-5s true_res=%-10s ||u||=%s\n" \
           "$R" "$NP" "$lab" "${it:-NA}" "${tr_:-NA}" "${un:-NA}"
  done
done
done
echo "wrote $OUT"
