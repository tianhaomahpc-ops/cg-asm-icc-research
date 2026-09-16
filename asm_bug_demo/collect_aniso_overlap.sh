#!/usr/bin/env bash
# collect_aniso_overlap.sh -- the delta-scaling test.
#
# Classical theory: C_0^2 <= C(1 + H/delta), and the partition of unity that
# proves it satisfies |grad chi| ~ 1/delta.  The multiplicity weight 1/sqrt(m_k)
# does NOT: it is a step at the seam whatever delta is.  So sASM's iteration
# count should SATURATE in overlap while -puramp / -puharm should keep paying.
# This script sweeps overlap far enough to see (or refute) that.
#
# usage: ./collect_aniso_overlap.sh [nx] [nranks] [fiber] [icc_level]
set -u
NX=${1:-48}; NP=${2:-4}; FIB=${3:-1,1,1}; L=${4:-0}
OUT="aniso_overlap_nx${NX}_n${NP}_L${L}.csv"
MPIRUN=${MPIRUN:-mpirun}
echo "ratio,method,O,L,iter,lmin,lmax,kappa" > "$OUT"
KSP="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 4000"
for R in 1 10 100; do
  echo "== r=$R =="
  for O in 0 1 2 3 4 5 6; do
    for M in "BASIC 0 -" "sASM 3 -" "ramp 7 -puramp" "harm 7 -puharm"; do
      set -- $M; lab=$1; sc=$2; fl=$3; [ "$fl" = "-" ] && fl=""
      out=$($MPIRUN --allow-run-as-root --oversubscribe -n "$NP" ./asm_demo \
            -nx "$NX" -fix_level 1 -scheme "$sc" $fl -aniso "$R" -fiber "$FIB" \
            -pc_type asm -pc_asm_type basic -pc_asm_overlap "$O" \
            -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
            $KSP -spectrum 2>&1)
      it=$(  sed -n 's/.*iters=\([0-9]*\).*/\1/p'    <<<"$out"|head -1)
      lmin=$(sed -n 's/.*lmin=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
      lmax=$(sed -n 's/.*lmax=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
      kap=$( sed -n 's/.*kappa=\([0-9.e+-]*\).*/\1/p'<<<"$out"|head -1)
      echo "$R,$lab,$O,$L,${it:-NA},${lmin:-NA},${lmax:-NA},${kap:-NA}" >> "$OUT"
      printf "  r=%-4s %-6s O=%d -> %-5s lmin=%s\n" "$R" "$lab" "$O" "${it:-NA}" "${lmin:-NA}"
    done
  done
done
echo "wrote $OUT"
