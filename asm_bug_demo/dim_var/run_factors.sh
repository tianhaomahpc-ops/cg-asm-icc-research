#!/bin/bash
# run_factors.sh -- parameter study: the influencing factors of ASM/sASM iteration count.
# Axes: H (subdomain count), L (ICC fill / omega), h (mesh resolution, weak scaling).
# Coarse space is a separate tool (schwarz_fem must support it) -- handled later.
set -u
cd "$(dirname "$0")"
mkdir -p results mesh
LOG=results/factors.log; : > "$LOG"
LAB=./schwarz_fem
run() { $LAB "$@" -sub_pc_factor_shift_type positive_definite 2>/dev/null | grep '\[RESULT\]'; }

# ----- Axis H : subdomain count (fixed problem size, 2D case a) ----------
echo "## AXIS H : subdomain count (2D case a, n=100 fixed, overlap 1)" | tee -a "$LOG"
for ns in 4 16 36 64 100 144; do
  python3 fem_build.py a 2 100 $ns >/dev/null 2>&1
  for part in box metis; do
    for sc in 0 3; do
      run -K mesh/casea_2D_K.petsc -b mesh/casea_2D_b.petsc -part mesh/casea_2D_${part}.is \
          -scheme $sc -overlap 1 -icc_levels 0 -measure_omega -tag "H_ns${ns}_${part}" | tee -a "$LOG"
    done
  done
done

# ----- Axis L : ICC fill level (2D cases a and d, fixed nsub=16, overlap 2) -----
echo "## AXIS L : ICC fill level (2D, overlap 2, nsub 16)" | tee -a "$LOG"
for cs in a d; do
  python3 fem_build.py $cs 2 70 16 >/dev/null 2>&1
  for sc in 0 3; do
    for L in 0 1 2 3; do
      run -K mesh/case${cs}_2D_K.petsc -b mesh/case${cs}_2D_b.petsc -part mesh/case${cs}_2D_box.is \
          -scheme $sc -overlap 2 -icc_levels $L -measure_omega -tag "L_${cs}_icc${L}" | tee -a "$LOG"
    done
    # exact local solve
    run -K mesh/case${cs}_2D_K.petsc -b mesh/case${cs}_2D_b.petsc -part mesh/case${cs}_2D_box.is \
        -scheme $sc -overlap 2 -exact -tag "L_${cs}_exact" | tee -a "$LOG"
  done
done

# ----- Axis h : weak scaling (fixed ~100 nodes/subdomain, 2D case a, overlap 1) -----
echo "## AXIS h : weak scaling (2D case a, ~100 nodes/subdomain, overlap 1)" | tee -a "$LOG"
#         (n, nsub) with n^2/nsub ~ const = 100
for pair in "40 16" "80 64" "120 144" "160 256"; do
  set -- $pair; n=$1; ns=$2
  python3 fem_build.py a 2 $n $ns >/dev/null 2>&1
  for sc in 0 3; do
    run -K mesh/casea_2D_K.petsc -b mesh/casea_2D_b.petsc -part mesh/casea_2D_box.is \
        -scheme $sc -overlap 1 -icc_levels 0 -measure_omega -tag "h_n${n}_ns${ns}" | tee -a "$LOG"
  done
done
echo "wrote $LOG"
