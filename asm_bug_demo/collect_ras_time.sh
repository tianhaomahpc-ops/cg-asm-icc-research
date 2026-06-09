#!/usr/bin/env bash
# RAS + GMRES: iter + fair solve-only time (warm, best-of-5) over O x L x restart.
# Same methodology as collect_time.sh so numbers are directly comparable to BASIC/sASM.
set -u
NX=48; NP=4; FL=1; REPS=5
OUT=ras_time_nx${NX}_n${NP}.csv
echo "ksp,O,L,restart,iter,min_time_s" > "$OUT"
K="-ksp_type gmres -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ # O L restart -> "iter mintime"
  local O="$1" L="$2" R="$3" it="" bt="" out t
  for r in $(seq 1 $REPS); do
    out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX -warmup $K \
          -ksp_gmres_restart $R -ksp_gmres_classicalgramschmidt \
          -pc_type asm -pc_asm_type restrict -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  echo "$it $bt"
}
echo "RAS+GMRES  O  L  restart | iter | min solve-only s (best of $REPS)"
for O in 1 2 3; do for L in 0 1 2; do for R in 30 60 120; do
  r=$(best $O $L $R); it=${r%% *}; bt=${r##* }
  printf "  O=%s L=%s r=%-3s | %-4s | %ss\n" "$O" "$L" "$R" "$it" "$bt"
  echo "gmres,$O,$L,$R,$it,$bt" >> "$OUT"
done; done; done
echo "--- high-restart time penalty check (O=2) ---"
for L in 0 1; do for R in 240 600; do
  r=$(best 2 $L $R); it=${r%% *}; bt=${r##* }
  printf "  O=2 L=%s r=%-3s | %-4s | %ss\n" "$L" "$R" "$it" "$bt"
  echo "gmres,2,$L,$R,$it,$bt" >> "$OUT"
done; done
echo "RAS_TIME_DONE -> $OUT"
