#!/usr/bin/env bash
# collect_aniso_ranks.sh -- is the controlling parameter H/delta rather than sigma?
#
# Hypothesis from the nx=48/4-rank sweep: the lambda_max over-count penalty is
# ~N_hat and sigma-blind, while the lambda_min payoff is set by how far delta
# layers reach relative to the subdomain size H.  With 4 ranks in 3D, H ~ 30h
# and delta <= 3, so H/delta ~ 10 and the payoff is only 1.2x -- the race is
# lost at every contrast.  More ranks shrink H at fixed delta, so the payoff
# should grow and the anomaly amplitude should fall.
#
# usage: ./collect_aniso_ranks.sh [nx] [icc_level]
set -u
NX=${1:-48}; L=${2:-0}
OUT="aniso_ranks_nx${NX}_L${L}.csv"
MPIRUN=${MPIRUN:-mpirun}
echo "ranks,ratio,method,O,iter,lmin,lmax,kappa,omega" > "$OUT"
KSP="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 4000"
for NP in 2 4 8 16; do
 for R in 1 100; do
  for O in 0 3; do
   for M in "BASIC 0 -" "sASM 3 -" "harm 7 -puharm"; do
    set -- $M; lab=$1; sc=$2; fl=$3; [ "$fl" = "-" ] && fl=""
    PROBE=""; [ "$sc" = "0" ] && PROBE="-probe_omega"
    out=$($MPIRUN --allow-run-as-root --oversubscribe -n "$NP" ./asm_demo \
          -nx "$NX" -fix_level 1 -scheme "$sc" $fl -aniso "$R" -fiber 1,1,1 \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap "$O" \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
          $KSP -spectrum $PROBE 2>&1)
    it=$(  sed -n 's/.*iters=\([0-9]*\).*/\1/p'    <<<"$out"|head -1)
    lmin=$(sed -n 's/.*lmin=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
    lmax=$(sed -n 's/.*lmax=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
    kap=$( sed -n 's/.*kappa=\([0-9.e+-]*\).*/\1/p'<<<"$out"|head -1)
    om=$(  sed -n 's/.*omega=\([0-9.e+-]*\).*/\1/p'<<<"$out"|head -1)
    echo "$NP,$R,$lab,$O,${it:-NA},${lmin:-NA},${lmax:-NA},${kap:-NA},${om:-NA}" >> "$OUT"
    printf "  P=%-3s r=%-4s %-6s O=%d -> %-5s lmin=%s\n" "$NP" "$R" "$lab" "$O" "${it:-NA}" "${lmin:-NA}"
   done
  done
 done
done
echo "wrote $OUT"
