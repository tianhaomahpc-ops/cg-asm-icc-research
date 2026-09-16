#!/usr/bin/env bash
# collect_aniso.sh -- the sigma-tensor sweep on the real MFEM driver.
#
# Reproduces, at production scale, the two headline curves measured in the
# self-contained harness (asm_bug_demo/aniso/):
#   (1) the overlap anomaly is a LOW-CONTRAST disease -- it dies as r grows;
#   (2) the sigma-harmonic PU (-puharm) beats multiplicity scaling, and the
#       gap widens with contrast.
#
# usage: ./collect_aniso.sh [nx] [nranks] [fiber]      e.g. ./collect_aniso.sh 48 4 1,1,0
set -u
NX=${1:-48}; NP=${2:-4}; FIB=${3:-1,1,0}
OUT="aniso_nx${NX}_n${NP}.csv"
echo "ratio,fiber,scheme,pu,O,L,iter,reason" > "$OUT"

KSP="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 \
     -ksp_max_it 4000 -ksp_converged_reason"

run () {   # $1 ratio  $2 scheme  $3 pu-flag(label)  $4 pu-flag(actual)  $5 O  $6 L
  local out it rs
  out=$(mpirun -n "$NP" ./asm_demo -nx "$NX" -fix_level 1 -scheme "$2" \
        -aniso "$1" -fiber "$FIB" $4 \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap "$5" \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$6" \
        $KSP -warmup 2>&1)
  it=$(echo "$out" | grep -o 'iters=[0-9]*' | head -1 | cut -d= -f2)
  rs=$(echo "$out" | grep -o 'CONVERGED_[A-Z_]*\|DIVERGED_[A-Z_]*' | head -1)
  echo "$1,$FIB,$2,$3,$5,$6,${it:-NA},${rs:-NA}" >> "$OUT"
  printf "  r=%-5s %-8s O=%d L=%d -> %s\n" "$1" "$3" "$5" "$6" "${it:-NA}"
}

for R in 1 2 5 10 20 50 100; do
  echo "== contrast r=$R =="
  for O in 0 1 2 3; do
    run "$R" 0 basic  ""                     "$O" 0     # PC_ASM_BASIC
    run "$R" 3 sASM   ""                     "$O" 0     # multiplicity scaling
    run "$R" 7 ramp   "-puramp"              "$O" 0     # NEW-A
    run "$R" 7 harm   "-puharm"              "$O" 0     # NEW-B
  done
done
echo "wrote $OUT"
