#!/usr/bin/env bash
# Sys2 (pure Neumann singular Laplace, ker=span{1}): four methods, optimal iter+time.
# warm solve-only, best-of-3. nx=48, 4 ranks.
set -u
NX=48; NP=4; FL=1; REPS=3
OUT=sys2_nx${NX}_n${NP}.csv
echo "method,O,L,extra,iter,min_time_s" > "$OUT"
K="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ # method scheme ksp asmtype O L "extra" extralabel
  local m="$1" S="$2" KT="$3" AT="$4" O="$5" L="$6" EX="$7" EL="$8" it="" bt="" out t
  for r in $(seq 1 $REPS); do
    out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX -pure_neumann -warmup \
          -ksp_type $KT $K -pc_type asm -pc_asm_type $AT -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L $EX 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  printf "%-12s O=%s L=%s %-8s | %-4s | %ss\n" "$m" "$O" "$L" "$EL" "$it" "$bt"
  echo "$m,$O,$L,$EL,$it,$bt" >> "$OUT"
}
echo "### Sys2 pure Neumann ###"
echo "-- BASIC --"
for O in 0 1 2; do for L in 0 1 2 3; do best BASIC 0 cg basic $O $L "" "-"; done; done
echo "-- sASM --"
for O in 0 1 2 3; do for L in 0 1 2 3; do best sASM 3 cg basic $O $L "" "-"; done; done
echo "-- sASM+Cheby --"
for O in 2 3; do for L in 1 2; do for D in 2 3; do best sASM+Cheby 4 cg basic $O $L "-localcheby $D" "d$D"; done; done; done
echo "-- RAS+GMRES --"
for O in 2 3; do for L in 1 2; do for R in 30 60 120; do best RAS+GMRES 0 gmres restrict $O $L "-ksp_gmres_restart $R -ksp_gmres_classicalgramschmidt" "r$R"; done; done; done
echo "SYS2_DONE -> $OUT"
