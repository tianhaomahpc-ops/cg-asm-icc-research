#!/usr/bin/env bash
# S4 (exact Cholesky vs ICC0) and S5 (BASIC vs sASM iter+time, best-of-3). nx=48, 4 ranks, fix_level=1.
set -u
NX=48; NP=4; FL=1
K="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason"
geti() { grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett() { grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }

# seed
mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX $K -ksp_type cg \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 0 -sub_ksp_type preonly \
  -sub_pc_type icc -sub_pc_factor_levels 0 >/dev/null 2>&1

echo "##### S4: BASIC + ICC(0) vs BASIC + exact Cholesky (CG) #####"
for O in 0 1 2; do
  i=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX -ksp_type cg $K \
       -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
       -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0 2>&1 | geti)
  c=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX -ksp_type cg $K \
       -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
       -sub_ksp_type preonly -sub_pc_type cholesky 2>&1 | geti)
  echo "O=$O  BASIC+ICC0=$i   BASIC+Cholesky=$c"
done

echo "##### S5: BASIC vs sASM  iter + wall time (best of 3), L=0 #####"
besttime() { # scheme O  -> echo "iter time"
  local S="$1" O="$2" it="" bt="" t
  for r in 1 2 3; do
    out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX -ksp_type cg $K \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0 2>&1)
    it=$(echo "$out" | geti); t=$(echo "$out" | gett)
    if [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; then bt=$t; fi
  done
  echo "$it $bt"
}
for O in 0 1 2; do
  b=$(besttime 0 $O); s=$(besttime 3 $O)
  echo "O=$O  BASIC iter/time = ${b}   sASM iter/time = ${s}"
done
echo "S4S5_DONE"
