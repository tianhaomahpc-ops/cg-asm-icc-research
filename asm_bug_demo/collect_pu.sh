#!/usr/bin/env bash
# scheme 7 (eps-PU weighted ASM) iteration sweep: eps x O x L. nx=48, 4 ranks, fix_level=1.
set -u
NX=48; NP=4; FL=1
OUT=pu_nx${NX}_n${NP}.csv
echo "eps,O,L,iter,reason,err" > "$OUT"
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000 -ksp_converged_reason"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
getr(){ grep -oE "due to [A-Z_]+ iterations" | head -1 | sed -E 's/due to ([A-Z_]+).*/\1/'; }
gete(){ grep -oE "\|\|u-u\*\|\|/\|\|u\*\|\|=[0-9.e+-]+" | head -1 | sed 's/.*=//'; }
for EPS in 0.0 0.1 0.25 0.5 0.7 1.0; do
 for O in 1 2 3; do for L in 0 1 2; do
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 7 -pueps $EPS -nx $NX $K \
        -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
  it=$(echo "$out"|geti); rr=$(echo "$out"|getr); er=$(echo "$out"|gete)
  printf "eps=%-4s O=%s L=%s | iter=%-4s %-14s err=%s\n" "$EPS" "$O" "$L" "${it:-NA}" "${rr:-NA}" "${er:-NA}"
  echo "$EPS,$O,$L,${it:-NA},${rr:-NA},${er:-NA}" >> "$OUT"
 done; done
done
echo "PU_DONE -> $OUT"
