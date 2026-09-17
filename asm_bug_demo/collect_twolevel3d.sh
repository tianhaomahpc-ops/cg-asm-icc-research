#!/usr/bin/env bash
# collect_twolevel3d.sh -- does the coarse level pay in 3D, and from how many
# subdomains on?
#
# At 4 MPI ranks a 4-function coarse space makes things WORSE: it lifts
# lambda_min ~1.7x but lifts lambda_max by about the same, so kappa is flat.
# Classical two-level theory says kappa <= C(1 + H/delta) INDEPENDENT of the
# number of subdomains, so the coarse level should start winning once there are
# enough of them.  This measures where that crossover is.
#
# usage: ./collect_twolevel3d.sh [nx] [icc_level] [ratio] [maxranks]
set -u
NX=${1:-48}; L=${2:-0}; R=${3:-1}; MAXP=${4:-16}
OUT="twolevel3d_nx${NX}_L${L}_r${R}.csv"
MPIRUN=${MPIRUN:-mpirun}
echo "ranks,ratio,method,O,iter,lmin,lmax,kappa" > "$OUT"
KSP="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 4000"

# Parallel arrays; flags are passed verbatim (word-split, never rewritten).
# The previous version ran them through `sed 's/ *- *//g'`, which ate the
# LEADING dash too, so -puharm arrived as an unrecognised `puharm` and every
# configuration silently degenerated to plain scheme 7.
LABELS=(  "sASM"  "harm"     "harm+coarse"        "mult+coarse" )
SCHEMES=( 3       7          7                    7             )
FLAGS=(   ""      "-puharm"  "-puharm -pucoarse"  "-pucoarse"   )

for NP in 2 4 8 16 32; do
  [ "$NP" -gt "$MAXP" ] && continue
  for k in "${!LABELS[@]}"; do
    lab=${LABELS[$k]}; sc=${SCHEMES[$k]}; fl=${FLAGS[$k]}
    out=$($MPIRUN --allow-run-as-root --oversubscribe -n "$NP" ./asm_demo \
          -nx "$NX" -fix_level 1 -scheme "$sc" $fl -aniso "$R" -fiber 1,1,1 \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
          $KSP -spectrum 2>&1)
    it=$(  sed -n 's/.*iters=\([0-9]*\).*/\1/p'    <<<"$out"|head -1)
    lmin=$(sed -n 's/.*lmin=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
    lmax=$(sed -n 's/.*lmax=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
    kap=$( sed -n 's/.*kappa=\([0-9.e+-]*\).*/\1/p'<<<"$out"|head -1)
    # provenance: what the binary itself reported about its configuration
    saw=$(grep -oE "shape = [a-z-]+|\[COARSE\]" <<<"$out" | tr '\n' ' ')
    echo "$NP,$R,$lab,2,${it:-NA},${lmin:-NA},${lmax:-NA},${kap:-NA}" >> "$OUT"
    printf "  P=%-3s %-12s flags[%-19s] banner[%-28s] -> iter=%-5s lmin=%-10s lmax=%s\n" \
           "$NP" "$lab" "$fl" "$saw" "${it:-NA}" "${lmin:-NA}" "${lmax:-NA}"
  done
done
echo "wrote $OUT"
