#!/usr/bin/env bash
# collect_twolevel3d.sh -- does the coarse level pay in 3D, and from how many
# subdomains on?
#
# At 4 MPI ranks the coarse space has only 4 basis functions and makes things
# WORSE (it lifts lambda_min 1.7x but lifts lambda_max just as much).  The
# question is whether that reverses once there are enough subdomains -- which
# is also the regime the anomaly actually matters in, since one-level Schwarz
# degrades like sqrt(#subdomains).
#
# usage: ./collect_twolevel3d.sh [nx] [icc_level] [ratio]
set -u
NX=${1:-48}; L=${2:-0}; R=${3:-1}
OUT="twolevel3d_nx${NX}_L${L}_r${R}.csv"
MPIRUN=${MPIRUN:-mpirun}
echo "ranks,ratio,method,O,iter,lmin,lmax,kappa" > "$OUT"
KSP="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 4000"
for NP in 2 4 8 16 32; do
  for M in "sASM              3 -|-" \
           "harm              7 -puharm|-" \
           "harm+coarse       7 -puharm|-pucoarse" \
           "mult+coarse       7 -pueps|1|-pucoarse"; do
    lab=$(awk '{print $1}' <<<"$M"); rest=${M#*$lab }
    sc=$(awk '{print $1}' <<<"$rest"); fl=$(awk '{print $2}' <<<"$rest" | tr '|' ' ')
    fl=$(sed 's/ *- *//g' <<<"$fl")
    out=$($MPIRUN --allow-run-as-root --oversubscribe -n "$NP" ./asm_demo \
          -nx "$NX" -fix_level 1 -scheme "$sc" $fl -aniso "$R" -fiber 1,1,1 \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap 2 \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
          $KSP -spectrum 2>&1)
    it=$(  sed -n 's/.*iters=\([0-9]*\).*/\1/p'    <<<"$out"|head -1)
    lmin=$(sed -n 's/.*lmin=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
    lmax=$(sed -n 's/.*lmax=\([0-9.e+-]*\).*/\1/p' <<<"$out"|head -1)
    kap=$( sed -n 's/.*kappa=\([0-9.e+-]*\).*/\1/p'<<<"$out"|head -1)
    echo "$NP,$R,$lab,2,${it:-NA},${lmin:-NA},${lmax:-NA},${kap:-NA}" >> "$OUT"
    printf "  P=%-3s %-16s -> iter=%-5s lmin=%-10s lmax=%s\n" "$NP" "$lab" "${it:-NA}" "${lmin:-NA}" "${lmax:-NA}"
  done
done
echo "wrote $OUT"
