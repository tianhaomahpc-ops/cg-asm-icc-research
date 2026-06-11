#!/usr/bin/env bash
set -u
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
run(){ mpirun -n 4 ./asm_demo -fix_level 1 -scheme 7 $2 -nx 48 $K \
       -pc_type asm -pc_asm_type basic -pc_asm_overlap $1 \
       -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $3 2>&1 | geti; }
echo "anchor1: q=1.0 must equal sASM 132/104/103 (O=0/1/2, L0)"
for O in 0 1 2; do echo "  q=1.0 O=$O : $(run $O "-pugrade 1.0" 0)"; done
echo "anchor2: O=1 graded q=0.7 must equal flat eps=0.7 (100 @ L0, 66 @ L2)"
echo "  q=0.7 O=1 L0 : $(run 1 "-pugrade 0.7" 0)"
echo "  q=0.7 O=1 L2 : $(run 1 "-pugrade 0.7" 2)"
echo "SANITY_DONE"
