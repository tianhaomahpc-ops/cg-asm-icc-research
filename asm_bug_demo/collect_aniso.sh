#!/usr/bin/env bash
# collect_aniso.sh -- the sigma-tensor sweep on the real MFEM/PETSc/MPI driver.
#
# Emits one CSV row per (contrast, overlap, method) with the iteration count and,
# for the BASIC baseline, the spectral decomposition that identifies WHICH of the
# two competing mechanisms drives the overlap anomaly:
#   lambda_max penalty (over-counting)  vs  lambda_min payoff (overlap reach)
# plus omega = max_i lambda_max(M_i^-1 A_i), the ICC sub-solve inexactness.
#
# usage: ./collect_aniso.sh [nx] [nranks] [fiber] [icc_level]
#        ./collect_aniso.sh 48 4 1,1,1 0
set -u
NX=${1:-48}; NP=${2:-4}; FIB=${3:-1,1,1}; L=${4:-0}
OUT="aniso_nx${NX}_n${NP}_L${L}.csv"
MPIRUN=${MPIRUN:-mpirun}
echo "ratio,fiber,method,O,L,iter,lmin,lmax,kappa,omega,kappa_sub,pnorm,relerr" > "$OUT"

KSP="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 4000"

run () {  # $1 ratio  $2 method-label  $3 scheme  $4 extra-flags  $5 O
  local out it lmin lmax kap om ks pn re PROBE
  # omega is a property of (A_i, ICC) alone, so probe it once, on the BASIC run
  PROBE=""; [ "$3" = "0" ] && PROBE="-probe_omega"
  out=$($MPIRUN --allow-run-as-root --oversubscribe -n "$NP" ./asm_demo \
        -nx "$NX" -fix_level 1 -scheme "$3" -aniso "$1" -fiber "$FIB" $4 \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap "$5" \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
        $KSP -spectrum $PROBE 2>&1)
  it=$(  sed -n 's/.*iters=\([0-9]*\).*/\1/p'          <<<"$out" | head -1)
  pn=$(  sed -n 's/.*final_pnorm=\([0-9.e+-]*\).*/\1/p' <<<"$out" | head -1)
  lmin=$(sed -n 's/.*lmin=\([0-9.e+-]*\).*/\1/p'        <<<"$out" | head -1)
  lmax=$(sed -n 's/.*lmax=\([0-9.e+-]*\).*/\1/p'        <<<"$out" | head -1)
  kap=$( sed -n 's/.*kappa=\([0-9.e+-]*\).*/\1/p'       <<<"$out" | head -1)
  re=$(  sed -n 's/.*rel_err=\([0-9.e+-]*\).*/\1/p'     <<<"$out" | head -1)
  om=$(sed -n 's/.*omega=\([0-9.e+-]*\).*/\1/p'     <<<"$out" | head -1)
  ks=$(sed -n 's/.*kappa_sub=\([0-9.e+-]*\).*/\1/p' <<<"$out" | head -1)
  echo "$1,$FIB,$2,$5,$L,${it:-NA},${lmin:-NA},${lmax:-NA},${kap:-NA},${om:-NA},${ks:-NA},${pn:-NA},${re:-NA}" >> "$OUT"
  printf "  r=%-5s %-6s O=%d -> iter=%-5s lmin=%-11s lmax=%-9s omega=%s\n" \
         "$1" "$2" "$5" "${it:-NA}" "${lmin:-NA}" "${lmax:-NA}" "${om:-NA}"
}

for R in 1 2 5 10 20 50 100; do
  echo "== contrast r=$R  (fiber $FIB, nx=$NX, $NP ranks, ICC($L)) =="
  for O in 0 1 2 3; do
    run "$R" BASIC 0 ""         "$O"
    run "$R" sASM  3 ""         "$O"
    run "$R" ramp  7 "-puramp"  "$O"
    run "$R" harm  7 "-puharm"  "$O"
  done
done
echo "wrote $OUT"
