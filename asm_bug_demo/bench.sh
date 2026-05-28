#!/usr/bin/env bash
# Sweep all 4 schemes x (overlap, ICC level), record iter count and KSP
# wall time.  Each cell is run REPEATS times; we report the median time
# (most robust against OS / interrupt noise on a laptop).
#
# Usage:  ./bench.sh [nx] [nranks] [repeats]   defaults 24, 4, 3
set -u
NX="${1:-24}"
NP="${2:-4}"
REPEATS="${3:-3}"
SOURCE="${4:-0}"

OUT="bench_nx${NX}_n${NP}_src${SOURCE}.csv"
echo "scheme,O,L,iters,median_time_s,rnorm" > "$OUT"

median() {
  # Portable median: sort numerically, pick middle line.
  local n
  n=$(printf "%s\n" "$@" | wc -l | tr -d ' ')
  local mid=$(( (n + 1) / 2 ))
  printf "%s\n" "$@" | sort -g | sed -n "${mid}p"
}

COMMON="-ksp_norm_type preconditioned \
        -ksp_initial_guess_nonzero false \
        -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
        -ksp_converged_reason \
        -sub_ksp_type preonly -sub_pc_type icc"

run_one() {
  local SCHEME=$1 O=$2 L=$3
  local ASM_OPT=""
  if [[ "$SCHEME" == "1" || "$SCHEME" == "2" ]]; then
    ASM_OPT=""   # scheme 1/2 internally set -pc_asm_type restrict
  else
    ASM_OPT="-pc_asm_type basic"
  fi
  # Suppress noisy [sASM step ...] debug prints from scheme 3 so the
  # solve sees no extra rank-0 stdout flushing while we are timing.
  mpirun -n "$NP" ./asm_demo \
      -fix_level 1 -scheme "$SCHEME" -nx "$NX" -source_type "$SOURCE" \
      $COMMON \
      -pc_type asm $ASM_OPT -pc_asm_overlap "$O" \
      -sub_pc_factor_levels "$L" \
      2>&1
}

printf "%-8s %-3s %-3s | %-6s | %-12s | %s\n" \
       "scheme" "O" "L" "iters" "median_time" "||r||/||b||"
echo "-----------------------------------------------------------------"
for S in 0 1 2 3; do
  for O in 0 1 2; do
    for L in 0 1 2; do
      ITERS=""
      RNORM=""
      TIMES=""
      for k in $(seq 1 "$REPEATS"); do
        OUT_RUN=$(run_one "$S" "$O" "$L")
        T=$(echo "$OUT_RUN" | grep -oE "time=[0-9.]+" | tail -1 \
              | sed -E 's/time=//')
        I=$(echo "$OUT_RUN" | grep -oE "iters=[0-9]+"  | tail -1 \
              | sed -E 's/iters=//')
        R=$(echo "$OUT_RUN" | grep -oE "\|\|r\|\|/\|\|b\|\|=[0-9.e+-]+" | tail -1 \
              | sed -E 's/.*=//')
        TIMES="$TIMES $T"
        ITERS="$I"
        RNORM="$R"
      done
      MED=$(median $TIMES)
      printf "%-8s %-3s %-3s | %-6s | %-12s | %s\n" \
             "$S" "$O" "$L" "$ITERS" "$MED" "$RNORM"
      echo "$S,$O,$L,$ITERS,$MED,$RNORM" >> "$OUT"
    done
  done
  echo "-----------------------------------------------------------------"
done
echo "wrote $OUT"
