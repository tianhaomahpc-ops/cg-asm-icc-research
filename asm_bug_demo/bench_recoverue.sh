#!/usr/bin/env bash
# Sweep ASM (scheme 0) and sASM (scheme 3) over (overlap, ICC level)
# on the all-Neumann recoverue P1 tet problem.  Each cell median over
# 3 repeats for time stability.  Compares iter count and time.
#
# Usage:  ./bench_recoverue.sh [nx] [ranks] [repeats]   default 24 4 3
set -u
NX="${1:-24}"
NP="${2:-4}"
REPEATS="${3:-5}"

OUT="bench_recoverue_nx${NX}_n${NP}.csv"
echo "scheme,O,L,iters,min_time_s,rnorm,err_rel" > "$OUT"

# Use min over repeats — more robust to OS scheduling noise than median.
mintime() {
  printf "%s\n" "$@" | sort -g | head -1
}

printf "%-8s %-3s %-3s | %-6s | %-12s | %-12s | %s\n" \
       "scheme" "O" "L" "iters" "min_time(s)" "||r||/||b||" "||u-Vm||/||Vm||"
echo "----------------------------------------------------------------------------"
for S in 0 3; do
  for O in 0 1 2; do
    for L in 0 1 2; do
      TIMES=""
      ITERS=""
      RNORM=""
      ERR=""
      for k in $(seq 1 "$REPEATS"); do
        OUT_RUN=$(mpirun -n "$NP" ./recoverue_demo \
                      -nx "$NX" -scheme "$S" -overlap "$O" -icc "$L" \
                      -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 \
                      -ksp_converged_reason 2>&1)
        I=$(echo "$OUT_RUN" | grep -oE "iters=[0-9]+" | tail -1 \
              | sed -E 's/iters=//')
        T=$(echo "$OUT_RUN" | grep -oE "time=[0-9.]+" | tail -1 \
              | sed -E 's/time=//')
        R=$(echo "$OUT_RUN" | grep -oE "\|\|r\|\|/\|\|b\|\|=[0-9.e+-]+" | tail -1 \
              | sed -E 's/.*=//')
        E=$(echo "$OUT_RUN" | grep -oE "Vm_0\|\|=[0-9.e+-]+" | tail -1 \
              | sed -E 's/.*=//')
        ITERS="$I"; RNORM="$R"; ERR="$E"
        TIMES="$TIMES $T"
      done
      MIN=$(mintime $TIMES)
      printf "%-8s %-3s %-3s | %-6s | %-12s | %-12s | %s\n" \
             "$S" "$O" "$L" "$ITERS" "$MIN" "$RNORM" "$ERR"
      echo "$S,$O,$L,$ITERS,$MIN,$RNORM,$ERR" >> "$OUT"
    done
  done
  echo "----------------------------------------------------------------------------"
done
echo "wrote $OUT"
