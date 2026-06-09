#!/usr/bin/env bash
# Fair solve-only (warm) timing: best-of-5 min time= for BASIC vs sASM. Sys3 nx=48 4 ranks.
set -u
NX=48; NP=4; FL=1; REPS=5
OUT=time_nx${NX}_n${NP}.csv
echo "scheme,O,L,iter,min_time_s" > "$OUT"
K="-ksp_type cg -ksp_norm_type preconditioned -ksp_initial_guess_nonzero false -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 3000"
geti(){ grep -oE "iters=[0-9]+" | head -1 | grep -oE "[0-9]+"; }
gett(){ grep -oE "time=[0-9.]+ s" | head -1 | grep -oE "[0-9.]+"; }
best(){ # scheme O L  -> "iter mintime"
  local S="$1" O="$2" L="$3" it="" bt="" out t
  for r in $(seq 1 $REPS); do
    out=$(mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX -warmup $K \
          -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
          -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L 2>&1)
    it=$(echo "$out"|geti); t=$(echo "$out"|gett)
    if [ -n "$t" ] && { [ -z "$bt" ] || awk "BEGIN{exit !($t<$bt)}"; }; then bt=$t; fi
  done
  echo "$it $bt"
}
echo "name   O  L | iter | min solve-only time (best of $REPS)"
sweep(){ local S="$1" name="$2"; shift 2
  for O in $1; do for L in $2; do
    r=$(best $S $O $L); it=${r%% *}; bt=${r##* }
    printf "%-5s O=%s L=%s | %-4s | %ss\n" "$name" "$O" "$L" "$it" "$bt"
    echo "$S,$O,$L,$it,$bt" >> "$OUT"
  done; done; }
echo "### BASIC (scheme 0) ###"
sweep 0 BASIC "0 1 2" "0 1 2 3 4"
echo "### sASM (scheme 3) ###"
sweep 3 sASM "0 1 2 3" "0 1 2 3 4"
echo "TIME_DONE -> $OUT"
