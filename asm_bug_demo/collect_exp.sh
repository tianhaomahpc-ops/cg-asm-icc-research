#!/usr/bin/env bash
# Experiments 1/2/3 sweep (run with: bash collect_exp.sh).  nx=48, 4 ranks, fix_level=1.
# Records iter, converged reason, ||r||/||b||, discretization error from asm_demo.
set -u
NX=48; NP=4; FL=1
OUT=exp_nx${NX}_n${NP}.csv
echo "exp,label,ksp,pc_asm_type,scheme,O,L,iter,reason,rnorm,err" > "$OUT"
KSPBASE="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason"

# run one config: args after the function name are extra mpirun flags.
# usage: run EXP LABEL KSP ASMTYPE SCHEME O L  <extra...>
run() {
  local exp="$1" label="$2" ksp="$3" asmt="$4" S="$5" O="$6" L="$7"; shift 7
  local out it rs rn er
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX \
        -ksp_type $ksp $KSPBASE \
        -pc_type asm -pc_asm_type $asmt -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L "$@" 2>&1)
  it=$(echo "$out" | grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+")
  rs=$(echo "$out" | grep -oE "due to [A-Z_]+ iterations" | head -1 | sed -E 's/due to ([A-Z_]+).*/\1/')
  rn=$(echo "$out" | grep -oE "\|\|r\|\|/\|\|b\|\|=[0-9.e+-]+" | head -1 | sed 's/.*=//')
  er=$(echo "$out" | grep -oE "\|\|u-u\*\|\|/\|\|u\*\|\|=[0-9.e+-]+" | head -1 | sed 's/.*=//')
  printf "%-22s O=%s L=%s | iter=%-5s %-16s rnorm=%-10s err=%s\n" "$label" "$O" "$L" "${it:-NA}" "${rs:-NA}" "${rn:-NA}" "${er:-NA}"
  echo "$exp,$label,$ksp,$asmt,$S,$O,$L,${it:-NA},${rs:-NA},${rn:-NA},${er:-NA}" >> "$OUT"
}

echo "############ seed matrix dump ############"
mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX $KSPBASE -ksp_type cg \
   -pc_type asm -pc_asm_type basic -pc_asm_overlap 0 -sub_ksp_type preonly \
   -sub_pc_type icc -sub_pc_factor_levels 0 >/dev/null 2>&1

echo "############ EXP 1: RAS + ICC(0), GMRES vs CG (L=0) ############"
for O in 0 1 2; do
  run exp1 "BASIC+CG (ref)"    cg    basic    0 $O 0
  run exp1 "RAS+GMRES"         gmres restrict 0 $O 0
  run exp1 "RAS+CG"            cg    restrict 0 $O 0
  run exp1 "sASM+CG (ref)"     cg    basic    3 $O 0
done

echo "############ EXP 2: D^-1 BASIC D^-1 (sym) vs BASIC vs sASM, full (O,L) ############"
for O in 0 1 2; do for L in 0 1 2; do
  run exp2 "BASIC (scheme0)"        cg basic 0 $O $L
  run exp2 "sASM  D^-1/2 (scheme3)" cg basic 3 $O $L
  run exp2 "exp2  D^-1   (scheme5)" cg basic 5 $O $L
done; done

echo "############ EXP 3: D^-1 BASIC (non-sym), GMRES vs CG (L=0) ############"
for O in 0 1 2; do
  run exp3 "exp3 D^-1 BASIC +GMRES" gmres basic 6 $O 0
  run exp3 "exp3 D^-1 BASIC +CG"    cg    basic 6 $O 0
  run exp3 "RAS+GMRES (compare)"    gmres restrict 0 $O 0
  run exp3 "sASM+CG (compare)"      cg    basic 3 $O 0
done
echo "############ DONE -> $OUT ############"
