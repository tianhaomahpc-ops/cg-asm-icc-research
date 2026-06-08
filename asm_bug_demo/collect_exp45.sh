#!/usr/bin/env bash
# Experiments 4 (GMRES restart/orthog) & 5 (FCG).  nx=48, 4 ranks, fix_level=1.
set -u
NX=48; NP=4; FL=1
OUT=exp45_nx${NX}_n${NP}.csv
echo "exp,label,ksp,pc_asm_type,scheme,O,restart,orthog,fcg_mmax,iter,reason,err" > "$OUT"
KSPBASE="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason"

# run EXP LABEL KSP ASMTYPE SCHEME O <extra petsc flags...>
run() {
  local exp="$1" label="$2" ksp="$3" asmt="$4" S="$5" O="$6"; shift 6
  local rs=30 og=cgs mm="-"
  for a in "$@"; do
    case "$a" in
      -ksp_gmres_restart) ;; *[0-9]) ;;
    esac
  done
  # parse restart / orthog / mmax for the csv label (best-effort)
  local flags="$*"
  [[ "$flags" == *"-ksp_gmres_restart "* ]] && rs=$(echo "$flags" | sed -E 's/.*-ksp_gmres_restart ([0-9]+).*/\1/')
  [[ "$flags" == *"modifiedgramschmidt"* ]] && og=mgs
  [[ "$flags" == *"-ksp_fcg_mmax "* ]] && mm=$(echo "$flags" | sed -E 's/.*-ksp_fcg_mmax ([0-9]+).*/\1/')
  local out it rr er
  out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX \
        -ksp_type $ksp $KSPBASE \
        -pc_type asm -pc_asm_type $asmt -pc_asm_overlap $O \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0 "$@" 2>&1)
  it=$(echo "$out" | grep -oE "iterations [0-9]+" | head -1 | grep -oE "[0-9]+")
  rr=$(echo "$out" | grep -oE "due to [A-Z_]+ iterations" | head -1 | sed -E 's/due to ([A-Z_]+).*/\1/')
  er=$(echo "$out" | grep -oE "\|\|u-u\*\|\|/\|\|u\*\|\|=[0-9.e+-]+" | head -1 | sed 's/.*=//')
  printf "%-26s O=%s rst=%-4s %-4s mmax=%-4s | iter=%-5s %-14s err=%s\n" "$label" "$O" "$rs" "$og" "$mm" "${it:-NA}" "${rr:-NA}" "${er:-NA}"
  echo "$exp,$label,$ksp,$asmt,$S,$O,$rs,$og,$mm,${it:-NA},${rr:-NA},${er:-NA}" >> "$OUT"
}

echo "############ seed ############"
mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX $KSPBASE -ksp_type cg \
  -pc_type asm -pc_asm_type basic -pc_asm_overlap 0 -sub_ksp_type preonly \
  -sub_pc_type icc -sub_pc_factor_levels 0 >/dev/null 2>&1

echo "############ EXP 4: RAS+GMRES restart x orthogonalization ############"
for O in 0 1 2; do
  for R in 30 60 120 240 600; do
    run exp4 "RAS+GMRES" gmres restrict 0 $O -ksp_gmres_restart $R -ksp_gmres_classicalgramschmidt
    run exp4 "RAS+GMRES" gmres restrict 0 $O -ksp_gmres_restart $R -ksp_gmres_modifiedgramschmidt
  done
done
echo "---- secondary: scheme6 (D^-1 BASIC) +GMRES restart, O=1,2 ----"
for O in 1 2; do for R in 30 600; do
  run exp4 "exp3-D^-1+GMRES" gmres basic 6 $O -ksp_gmres_restart $R -ksp_gmres_classicalgramschmidt
done; done

echo "############ EXP 5: RAS + FCG (flexible CG) ############"
for O in 0 1 2; do
  run exp5 "RAS+FCG"            fcg   restrict 0 $O
  run exp5 "RAS+CG (ref,fails)" cg    restrict 0 $O
  run exp5 "RAS+GMRES (ref)"    gmres restrict 0 $O
  run exp5 "sASM+CG (ref)"      cg    basic    3 $O
done
echo "---- FCG mmax sweep at O=2 ----"
for M in 30 60 120 600; do
  run exp5 "RAS+FCG" fcg restrict 0 2 -ksp_fcg_mmax $M
done
echo "---- scheme6 (D^-1 BASIC) + FCG, O=1,2 ----"
for O in 1 2; do
  run exp5 "exp3-D^-1+FCG" fcg basic 6 $O
done
echo "EXP45_DONE -> $OUT"
