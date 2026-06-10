#!/usr/bin/env bash
# All-Dirichlet Laplace (u=0 on all 6 faces): iteration count vs overlap.
# BASIC O=0..5 x L=0..5, sASM O=0..5 x L=0..2, exact-Cholesky reference. nx=48, 4 ranks.
set -u
NX=48; NP=4; FL=1
OUT=sysD_nx${NX}_n${NP}.csv
echo "method,O,L,iter,reason" > "$OUT"
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
getr(){ grep -oE "due to [A-Z_]+ iterations" | head -1 | sed -E 's/due to ([A-Z_]+).*/\1/'; }
run(){ # method scheme O L subpc
  local m="$1" S="$2" O="$3" L="$4" SP="$5" out it rr
  if [ "$SP" = "cholesky" ]; then
    out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX -all_dirichlet $K \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type cholesky 2>&1)
  else
    out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX -all_dirichlet $K \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  fi
  it=$(echo "$out"|geti); rr=$(echo "$out"|getr)
  printf "%-10s O=%s L=%-3s | iter=%-4s %s\n" "$m" "$O" "$L" "${it:-NA}" "${rr:-NA}"
  echo "$m,$O,$L,${it:-NA},${rr:-NA}" >> "$OUT"
}
echo "-- BASIC + ICC --"
for L in 0 1 2 3 4 5; do for O in 0 1 2 3 4 5; do run BASIC 0 $O $L icc; done; done
echo "-- sASM + ICC --"
for L in 0 1 2; do for O in 0 1 2 3 4 5; do run sASM 3 $O $L icc; done; done
echo "-- BASIC + exact Cholesky --"
for O in 0 1 2 3 5; do run Chol 0 $O - cholesky; done
echo "SYSD_DONE -> $OUT"
