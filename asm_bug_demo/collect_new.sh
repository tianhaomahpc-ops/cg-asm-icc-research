#!/usr/bin/env bash
# New-method sweeps: scheme 8 (sMRAS) and scheme 7 graded+cheby. nx=48, 4 ranks.
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
getr(){ grep -oE "due to [A-Z_]+" | head -1 | sed 's/due to //'; }
run(){ local S="$1" EX="$2" O="$3" L="$4" out
  out=$(mpirun -n 4 ./asm_demo -fix_level 1 -scheme $S $EX -nx 48 $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  printf "%s\n" "$(echo "$out"|geti) $(echo "$out"|getr)"; }
echo "## scheme 8 (sMRAS, theta=1): O x L ##"
for O in 1 2 3; do for L in 0 1 2; do
  printf "sMRAS O=%s L=%s | %s\n" "$O" "$L" "$(run 8 "" $O $L)"
done; done
echo "## scheme 7 graded+cheby: q x d x O x L ##"
for Q in 0.7 0.8; do for D in 2 3; do for O in 2 3; do for L in 0 1; do
  printf "g%s-d%s O=%s L=%s | %s\n" "$Q" "$D" "$O" "$L" "$(run 7 "-pugrade $Q -localcheby $D" $O $L)"
done; done; done; done
echo "NEW_DONE"
