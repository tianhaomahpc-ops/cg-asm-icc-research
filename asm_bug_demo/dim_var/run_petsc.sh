#!/bin/bash
# run_petsc.sh -- full publishable experiment matrix, REAL PETSc (schwarz_lab).
# All runs serial (-n 1) with explicit S^d subdomains for reproducibility.
# Emits tagged [RESULT] lines to results/petsc_<exp>.log ; parse_petsc.py builds CSV+figures.
#
#   ./run_petsc.sh A   # dimensional, constant coef (BASIC/sASM x exact/ICC0 x overlap)
#   ./run_petsc.sh C    # variable coefficient (fields @1e4 + contrast sweep)
#   ./run_petsc.sh all
set -u
cd "$(dirname "$0")"
mkdir -p results
LAB=./schwarz_lab
run() { $LAB "$@" 2>/dev/null | grep '\[RESULT\]'; }

expA() {
  local out=results/petsc_A.log; : > "$out"
  echo "# Experiment A: dimensional, constant coefficient" | tee "$out"
  # dim : nx : S : overlaps
  for spec in "1:257:4:0 1 2 4 8 16" "2:49:4:0 1 2 3 4 6" "3:25:4:0 1 2 3 4 5"; do
    IFS=':' read d M S OVS <<< "$spec"
    for sc in 0 3; do                       # 0=BASIC  3=sASM
      for ex in "" "-exact"; do             # ICC0 vs exact local
        for ov in $OVS; do
          run -dim $d -nx $M -S $S -overlap $ov -scheme $sc -icc_levels 0 $ex -measure_omega | tee -a "$out"
        done
      done
    done
  done
  echo "wrote $out"
}

expC() {
  local out=results/petsc_C.log; : > "$out"
  echo "# Experiment C: variable coefficient" | tee "$out"
  # C1: coefficient fields at contrast 1e4, BASIC vs sASM, overlap sweep, 2D & 3D
  for spec in "2:49:4:0 1 2 3 4" "3:25:4:0 1 2 3 4"; do
    IFS=':' read d M S OVS <<< "$spec"
    for coef in const smooth layers layers_unaligned checker; do
      rho=10000; [ "$coef" = const ] && rho=1; [ "$coef" = smooth ] && rho=1
      for sc in 0 3; do
        for ov in $OVS; do
          run -dim $d -nx $M -S $S -overlap $ov -scheme $sc -icc_levels 0 \
              -coef $coef -contrast $rho -measure_omega \
              -sub_pc_factor_shift_type positive_definite | tee -a "$out"
        done
      done
    done
  done
  # C2: contrast sweep at fixed (dim, overlap=2), aligned vs unaligned jumps
  echo "# C2 contrast sweep" | tee -a "$out"
  for d in 2 3; do
    M=49; [ "$d" = 3 ] && M=25
    for coef in layers layers_unaligned checker; do
      for sc in 0 3; do
        for rho in 1 10 100 1000 10000; do
          run -dim $d -nx $M -S 4 -overlap 2 -scheme $sc -icc_levels 0 \
              -coef $coef -contrast $rho -measure_omega \
              -sub_pc_factor_shift_type positive_definite | tee -a "$out"
        done
      done
    done
  done
  echo "wrote $out"
}

case "${1:-A}" in
  A) expA ;;
  C) expC ;;
  all) expA; expC ;;
  *) echo "usage: $0 A|C|all" ;;
esac
