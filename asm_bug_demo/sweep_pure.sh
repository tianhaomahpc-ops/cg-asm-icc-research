#!/usr/bin/env bash
# Same sweep, but using the pure-PETSc driver pure_petsc_demo.
# Usage:
#   ./sweep_pure.sh <scheme> [nx] [nranks]
set -u
SCHEME="${1:-0}"
NX="${2:-24}"
NP="${3:-4}"

OUT="results_pure_scheme${SCHEME}_nx${NX}_n${NP}.csv"
echo "O,L,iters,reason,nz_used,pnorm,time_s" > "$OUT"

if [[ "$SCHEME" == "1" || "$SCHEME" == "2" ]]; then
    ASM_OPT=""
else
    ASM_OPT="-pc_asm_type basic"
fi

printf "[pure PETSc]  scheme=%s  nx=%s  ranks=%s\n" "$SCHEME" "$NX" "$NP"
printf "%-3s %-3s | %-6s | %-26s | %s\n" "O" "L" "iters" "reason" "time(s)"
echo "------------------------------------------------------------"

COMMON_KSP="-ksp_norm_type preconditioned \
            -ksp_initial_guess_nonzero false \
            -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
            -ksp_converged_reason"

for O in 0 1 2; do
  for L in 0 1 2; do
    LOG=$(mktemp -t purelog.XXXXXX)
    mpirun -n "$NP" ./pure_petsc_demo \
        -scheme "$SCHEME" -nx "$NX" \
        $COMMON_KSP \
        -pc_type asm $ASM_OPT -pc_asm_overlap "$O" \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
        > "$LOG" 2>&1
    ITERS=$(grep -E 'iterations [0-9]+' "$LOG" | tail -1 \
                | sed -E 's/.*iterations ([0-9]+).*/\1/')
    REASON=$(grep -E 'CONVERGED|DIVERGED' "$LOG" | tail -1 \
                | sed -E 's/.*due to ([A-Z_]+) iterations.*/\1/')
    NZ_USED=$(grep -E 'nz_used=[0-9]+' "$LOG" | tail -1 \
                | sed -E 's/.*nz_used=([0-9]+).*/\1/')
    PNORM=$(grep -E '\[RESULT\]' "$LOG" | tail -1 \
                | sed -E 's/.*pnorm=([^ ]+) .*/\1/')
    TIME=$(grep -E '\[RESULT\]' "$LOG" | tail -1 \
                | sed -E 's/.*time=([^ ]+) s/\1/')
    ITERS="${ITERS:-NA}"; REASON="${REASON:-NA}"
    NZ_USED="${NZ_USED:-NA}"; PNORM="${PNORM:-NA}"; TIME="${TIME:-NA}"
    printf "%-3s %-3s | %-6s | %-26s | %s\n" "$O" "$L" "$ITERS" "$REASON" "$TIME"
    echo "$O,$L,$ITERS,$REASON,$NZ_USED,$PNORM,$TIME" >> "$OUT"
    rm -f "$LOG"
  done
done
echo "wrote $OUT"
