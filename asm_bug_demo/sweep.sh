#!/usr/bin/env bash
# Sweep (overlap O) x (ICC level L) for a given (-fix_level F, -scheme S).
# Output: CSV file + pretty table to stdout.
#
# Usage:
#   ./sweep.sh <fix_level> <scheme> [nx] [nranks]
#
#   scheme 0 : CG    + ASM BASIC                       (-pc_asm_type basic)
#   scheme 1 : GMRES + ASM RESTRICT (RAS, non-sym)
#   scheme 2 : BCGS  + ASM RESTRICT (RAS, non-sym)
#   scheme 3 : CG    + sASM PCSHELL (D^-1/2 BASIC D^-1/2)
#
# Defaults: nx=24, nranks=4
set -u
LEVEL="${1:-0}"
SCHEME="${2:-0}"
NX="${3:-24}"
NP="${4:-4}"

OUT="results_fix${LEVEL}_scheme${SCHEME}_nx${NX}_n${NP}.csv"
echo "O,L,iters,reason,nz_used,nz_unneeded,pnorm,time_s" > "$OUT"

# For schemes 1 and 2 we must NOT pass -pc_asm_type basic (the code
# overrides it via PetscOptionsSetValue to "restrict" anyway).  For
# scheme 3 we ALSO leave -pc_asm_type basic but the outer PC gets
# replaced by PCSHELL, so it does not matter.  Simplest: pass it for
# scheme 0 and 3, omit for 1 and 2 to keep PETSc's options stack clean.
if [[ "$SCHEME" == "1" || "$SCHEME" == "2" ]]; then
    ASM_OPT=""        # InjectSchemeOptions sets -pc_asm_type restrict
else
    ASM_OPT="-pc_asm_type basic"
fi

printf "fix_level=%s  scheme=%s  nx=%s  ranks=%s\n" \
       "$LEVEL" "$SCHEME" "$NX" "$NP"
printf "%-3s %-3s | %-6s | %-26s | %s\n" \
       "O" "L" "iters" "reason" "time(s)"
echo "------------------------------------------------------------"

# Common KSP options.  We use preconditioned norm and zero initial
# guess so per-(O,L) tolerances are strictly comparable.
COMMON_KSP="-ksp_norm_type preconditioned \
            -ksp_initial_guess_nonzero false \
            -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1000 \
            -ksp_converged_reason"

for O in 0 1 2; do
  for L in 0 1 2; do
    LOG=$(mktemp -t asmlog.XXXXXX)
    mpirun -n "$NP" ./asm_demo \
        -fix_level "$LEVEL" -scheme "$SCHEME" -nx "$NX" \
        $COMMON_KSP \
        -pc_type asm $ASM_OPT -pc_asm_overlap "$O" \
        -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels "$L" \
        > "$LOG" 2>&1
    ITERS=$(grep -E 'iterations [0-9]+' "$LOG" | tail -1 \
                | sed -E 's/.*iterations ([0-9]+).*/\1/')
    REASON=$(grep -E 'CONVERGED|DIVERGED' "$LOG" | tail -1 \
                | sed -E 's/.*due to ([A-Z_]+) iterations.*/\1/')
    NZ_USED=$(grep -E 'nz_used = [0-9]+' "$LOG" | tail -1 \
                | sed -E 's/.*nz_used = ([0-9]+).*/\1/')
    NZ_UN=$(grep -E 'nz_unneeded = [0-9]+' "$LOG" | tail -1 \
                | sed -E 's/.*nz_unneeded = ([0-9]+).*/\1/')
    PNORM=$(grep -E '\[RESULT\]' "$LOG" | tail -1 \
                | sed -E 's/.*final_pnorm=([^ ]+) .*/\1/')
    TIME=$(grep -E '\[RESULT\]' "$LOG" | tail -1 \
                | sed -E 's/.*time=([^ ]+) s/\1/')
    ITERS="${ITERS:-NA}"; REASON="${REASON:-NA}"
    NZ_USED="${NZ_USED:-NA}"; NZ_UN="${NZ_UN:-NA}"
    PNORM="${PNORM:-NA}"; TIME="${TIME:-NA}"
    printf "%-3s %-3s | %-6s | %-26s | %s\n" "$O" "$L" "$ITERS" "$REASON" "$TIME"
    echo "$O,$L,$ITERS,$REASON,$NZ_USED,$NZ_UN,$PNORM,$TIME" >> "$OUT"
    rm -f "$LOG"
  done
done
echo "wrote $OUT"
