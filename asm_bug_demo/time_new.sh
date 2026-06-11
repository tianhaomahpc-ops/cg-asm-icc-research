#!/usr/bin/env bash
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ local S="$1" EX="$2" O="$3" L="$4" KT="${5:-cg}" AT="${6:-basic}" it="" bt="" out t
  for r in 1 2 3 4 5; do
    out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme $S $EX -nx 48 -warmup \
          -ksp_type $KT -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
          -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 \
          -pc_type asm -pc_asm_type $AT -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  echo "$it $bt"
}
echo "== leaders (solve-only, warm, best-of-5) =="
echo "sMRAS      O3 L2      : $(best 8 "" 3 2)"
echo "sMRAS      O2 L2      : $(best 8 "" 2 2)"
echo "g0.8-d2    O2 L1      : $(best 7 "-pugrade 0.8 -localcheby 2" 2 1)"
echo "g0.8-d2    O3 L1      : $(best 7 "-pugrade 0.8 -localcheby 2" 3 1)"
echo "g0.8-d3    O3 L1      : $(best 7 "-pugrade 0.8 -localcheby 3" 3 1)"
echo "g0.7-d3    O3 L1      : $(best 7 "-pugrade 0.7 -localcheby 3" 3 1)"
echo "== references, same batch =="
echo "scheme4 d2 O2 L1      : $(best 4 "" 2 1)"
echo "graded q0.8 O3 L2     : $(best 7 "-pugrade 0.8" 3 2)"
echo "RAS+GMRES r60 O3 L2   : $(best 0 "-ksp_gmres_restart 60 -ksp_gmres_classicalgramschmidt" 3 2 gmres restrict)"
echo "TIME_NEW_DONE"
