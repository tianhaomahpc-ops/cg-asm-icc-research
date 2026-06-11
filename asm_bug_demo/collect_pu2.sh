#!/usr/bin/env bash
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
echo "## fine eps sweep near the optimum ##"
for EPS in 0.6 0.8 0.9; do for O in 2 3; do for L in 1 2; do
  out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 -pueps $EPS -nx 48 $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  printf "eps=%-4s O=%s L=%s | iter=%s\n" "$EPS" "$O" "$L" "$(echo "$out"|geti)"
done; done; done
echo "## solve-only timing (warm, best-of-5): scheme7 eps=0.7 vs refs ##"
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ local S="$1" O="$2" L="$3" EX="$4" it="" bt="" out t
  for r in 1 2 3 4 5; do
    out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme $S -nx 48 -warmup $K $EX \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  echo "$it $bt"
}
for cfg in "2 1" "2 2" "3 1" "3 2"; do
  set -- $cfg; O=$1; L=$2
  r7=$(best 7 $O $L "-pueps 0.7"); r3=$(best 3 $O $L "")
  echo "O=$O L=$L | scheme7(e0.7) iter/time = $r7 | sASM iter/time = $r3"
done
echo "PU2_DONE"
