#!/usr/bin/env bash
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ local EX="$1" O="$2" L="$3" it="" bt="" out t
  for r in 1 2 3 4 5; do
    out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 $EX -nx 48 -warmup $K \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  echo "$it $bt"
}
echo "graded q=0.8 O3L2 : $(best "-pugrade 0.8" 3 2)"
echo "graded q=0.7 O2L2 : $(best "-pugrade 0.7" 2 2)"
echo "flat  e=0.7 O3L2 : $(best "-pueps 0.7" 3 2)"
echo "sASM(q=1)   O3L2 : $(best "-pugrade 1.0" 3 2)"
echo "TIME_GRADE_DONE"
