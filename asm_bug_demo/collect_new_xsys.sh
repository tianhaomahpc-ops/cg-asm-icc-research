#!/usr/bin/env bash
# New methods (graded+Cheby scheme 7, sMRAS scheme 8) on Sys1 (dt=1e-2) and Sys2 (pure Neumann).
# Phase 1: iteration sweeps.  Phase 2: same-batch timing at optima + refs.  nx=48, 4 ranks.
set -u
K="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
getr(){ grep -oE "due to [A-Z_]+" | head -1 | sed 's/due to //'; }
getres(){ grep -oE "\|\|r\|\|/\|\|b\|\|=[0-9.e+-]+" | head -1 | sed 's/.*=//'; }
run(){ # SYSEX SCHEME SCHEX KT AT O L  -> "iter reason"
  local SYSEX="$1" S="$2" EX="$3" KT="$4" AT="$5" O="$6" L="$7" out
  out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme $S $SYSEX $EX -nx 48 -ksp_type $KT $K \
        -pc_type asm -pc_asm_type $AT -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  echo "$(echo "$out"|geti) $(echo "$out"|getr) $(echo "$out"|getres)"; }

for SYS in "SYS1:-dt 1e-2" "SYS2:-pure_neumann"; do
  NAME=${SYS%%:*}; SX=${SYS#*:}
  echo "######## $NAME ########"
  echo "-- graded+Cheby --"
  for Q in 0.7 0.8; do for D in 2 3; do for O in 2 3; do for L in 0 1; do
    printf "%s g%s-d%s O=%s L=%s | %s\n" "$NAME" "$Q" "$D" "$O" "$L" "$(run "$SX" 7 "-pugrade $Q -localcheby $D" cg basic $O $L)"
  done; done; done; done
  echo "-- sMRAS --"
  for O in 1 2 3; do for L in 0 1 2; do
    printf "%s sMRAS O=%s L=%s | %s\n" "$NAME" "$O" "$L" "$(run "$SX" 8 "" cg basic $O $L)"
  done; done
done
echo "XSYS_DONE"
