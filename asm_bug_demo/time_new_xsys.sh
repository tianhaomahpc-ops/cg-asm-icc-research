#!/usr/bin/env bash
# Same-batch solve-only timing (warm, best-of-5) of new-method optima + refs on Sys1/Sys2.
set -u
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ # SYSEX SCHEME SCHEX KT AT O L
  local SX="$1" S="$2" EX="$3" KT="$4" AT="$5" O="$6" L="$7" it="" bt="" out t
  for r in 1 2 3 4 5; do
    out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme $S $SX $EX -nx 48 -warmup \
          -ksp_type $KT -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
          -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 \
          -pc_type asm -pc_asm_type $AT -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  echo "$it $bt"
}
echo "== SYS1 (dt=1e-2) =="
echo "g0.7-d2 O3L1   : $(best "-dt 1e-2" 7 "-pugrade 0.7 -localcheby 2" cg basic 3 1)"
echo "g0.7-d3 O3L1   : $(best "-dt 1e-2" 7 "-pugrade 0.7 -localcheby 3" cg basic 3 1)"
echo "sMRAS   O3L2   : $(best "-dt 1e-2" 8 "" cg basic 3 2)"
echo "sMRAS   O2L1   : $(best "-dt 1e-2" 8 "" cg basic 2 1)"
echo "scheme4 d2 O3L1: $(best "-dt 1e-2" 4 "" cg basic 3 1)"
echo "sASM    O2L1   : $(best "-dt 1e-2" 3 "" cg basic 2 1)"
echo "RAS+GMRES O2L1 : $(best "-dt 1e-2" 0 "-ksp_gmres_restart 30 -ksp_gmres_classicalgramschmidt" gmres restrict 2 1)"
echo "== SYS2 (pure Neumann) =="
echo "g0.8-d3 O3L1   : $(best "-pure_neumann" 7 "-pugrade 0.8 -localcheby 3" cg basic 3 1)"
echo "g0.7-d2 O3L1   : $(best "-pure_neumann" 7 "-pugrade 0.7 -localcheby 2" cg basic 3 1)"
echo "sMRAS   O3L2   : $(best "-pure_neumann" 8 "" cg basic 3 2)"
echo "sMRAS   O2L1   : $(best "-pure_neumann" 8 "" cg basic 2 1)"
echo "scheme4 d2 O2L1: $(best "-pure_neumann" 4 "" cg basic 2 1)"
echo "sASM    O2L2   : $(best "-pure_neumann" 3 "" cg basic 2 2)"
echo "RAS+GMRES O2L2 : $(best "-pure_neumann" 0 "-ksp_gmres_restart 60 -ksp_gmres_classicalgramschmidt" gmres restrict 2 2)"
echo "TIME_XSYS_DONE"
