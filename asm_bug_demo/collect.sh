#!/usr/bin/env bash
# Authoritative, single-code-state data collection for the report.
# Robust against laptop contention: each run writes its own log file,
# we parse the PETSc-internal "CONVERGED ... iterations N" count, and
# pace runs with a short sleep.  nx=48, 4 ranks, fix_level=1.
set -u
NX=48; NP=4; FL=1
LOGD=$(mktemp -d -t asmcollect.XXXXXX)
OUT=collect_nx${NX}_n${NP}.csv
echo "kind,scheme,O,L,impl,iters,reductions,time_s" > "$OUT"

KSP="-ksp_norm_type preconditioned -ksp_initial_guess_nonzero false \
     -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 2000 -ksp_converged_reason"

# parse "iterations N" (PETSc internal, robust) from a log file
pit() { grep -oE "iterations [0-9]+" "$1" | head -1 | grep -oE "[0-9]+"; }
ptime() { grep -oE "time=[0-9.]+" "$1" | tail -1 | sed 's/time=//'; }
pred() { grep "MPI Reductions:" "$1" | head -1 | sed -E 's/.*MPI Reductions:[[:space:]]*([0-9.e+]+).*/\1/'; }

asmrun() { # scheme O L extra... -> logfile echoed
  local S=$1 O=$2 L=$3; shift 3
  local lg="$LOGD/asm_s${S}_o${O}_l${L}_$$_${RANDOM}.log"
  mpirun -n $NP ./asm_demo -fix_level $FL -scheme $S -nx $NX $KSP \
     -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
     -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L \
     "$@" > "$lg" 2>&1
  echo "$lg"
}
loadrun() { # scheme O L -> logfile echoed (pure_petsc_load)
  local S=$1 O=$2 L=$3
  local lg="$LOGD/load_s${S}_o${O}_l${L}_$$_${RANDOM}.log"
  mpirun -n $NP ./pure_petsc_load -nx $NX -scheme $S $KSP \
     -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
     -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels $L \
     > "$lg" 2>&1
  echo "$lg"
}

echo "[collect] seeding matrix dump (nx=$NX, fix_level=$FL)"
lg=$(asmrun 0 0 0); sleep 0.5

echo "[collect] A: MFEM scheme 0/3/4 full (O,L) iter"
for S in 0 3 4; do for O in 0 1 2; do for L in 0 1 2; do
  lg=$(asmrun $S $O $L); it=$(pit "$lg")
  echo "iter,$S,$O,$L,mfem,${it:-NA},," >> "$OUT"; sleep 0.3
done; done; done

echo "[collect] B: pure-PETSc scheme 0/3/4 full (O,L) iter (cross-check)"
# re-seed dump from a fresh asm run so the loaded matrix is current
lg=$(asmrun 0 0 0); sleep 0.5
for S in 0 3 4; do for O in 0 1 2; do for L in 0 1 2; do
  lg=$(loadrun $S $O $L); it=$(pit "$lg")
  echo "iter,$S,$O,$L,petsc,${it:-NA},," >> "$OUT"; sleep 0.3
done; done; done

echo "[collect] C: experiment 2 — BASIC + exact Cholesky, O=0/1/2"
for O in 0 1 2; do
  lg="$LOGD/chol_o${O}.log"
  mpirun -n $NP ./asm_demo -fix_level $FL -scheme 0 -nx $NX $KSP \
     -pc_type asm -pc_asm_type basic -pc_asm_overlap $O \
     -sub_ksp_type preonly -sub_pc_type cholesky > "$lg" 2>&1
  it=$(pit "$lg")
  echo "chol,0,$O,-,mfem,${it:-NA},," >> "$OUT"; sleep 0.3
done

echo "[collect] D: experiment 5 — -log_view reductions, scheme 0/3/4, O=0/1/2, L=0"
for S in 0 3 4; do for O in 0 1 2; do
  lg=$(asmrun $S $O 0 -log_view)
  it=$(pit "$lg"); rd=$(pred "$lg")
  echo "reduc,$S,$O,0,mfem,${it:-NA},${rd:-NA}," >> "$OUT"; sleep 0.3
done; done

echo "[collect] E: wall-time median-of-3, scheme 0/3/4, O=1, L=0/1/2"
for S in 0 3 4; do for L in 0 1 2; do
  t1=$(ptime "$(asmrun $S 1 $L)"); sleep 0.3
  t2=$(ptime "$(asmrun $S 1 $L)"); sleep 0.3
  t3=$(ptime "$(asmrun $S 1 $L)"); sleep 0.3
  med=$(printf "%s\n%s\n%s\n" "$t1" "$t2" "$t3" | sort -g | sed -n '2p')
  lg=$(asmrun $S 1 $L); it=$(pit "$lg")
  echo "time,$S,1,$L,mfem,${it:-NA},,${med:-NA}" >> "$OUT"; sleep 0.3
done; done

echo "[collect] DONE -> $OUT"
rm -rf "$LOGD"
