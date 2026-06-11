#!/usr/bin/env bash
# Probe the remaining weight-shape freedom: graded q + normalizer exponent alpha.
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
run(){ mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 $2 -nx 48 $K \
       -pc_type asm -pc_asm_type basic -pc_asm_overlap $1 \
       -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $3 2>&1 | geti; }
echo "## graded q sweep (refs: flat e0.7 -> O2L0 98 O3L0 98 O2L2 62 O3L2 61; sASM 103/102/67/66; RAS-fullGMRES 92/92/57/51) ##"
for Q in 0.4 0.5 0.6 0.7 0.8 0.9; do for O in 2 3; do for L in 0 2; do
  printf "q=%-4s O=%s L=%s | iter=%s\n" "$Q" "$O" "$L" "$(run $O "-pugrade $Q" $L)"
done; done; done
echo "## alpha probe at eps=0.7 (alpha=1 refs: O3L0 98, O3L2 61) ##"
for A in 0.6 0.8 1.2 1.4; do for L in 0 2; do
  printf "alpha=%-4s O=3 L=%s | iter=%s\n" "$A" "$L" "$(run 3 "-pueps 0.7 -pualpha $A" $L)"
done; done
echo "PU3_DONE"
