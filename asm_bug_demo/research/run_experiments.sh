#!/usr/bin/env bash
# run_experiments.sh -- reproduce the ASM/sASM/two-level study in REPORT_research_zh.md.
# Container toolchain used: MFEM 4.9 (source build) + PETSc 3.19 + OpenMPI 4.1 + METIS,
# built as in the cardiac-sim-fakegeo notes. asm_demo = cube Laplace (Sys3 mixed BC;
# -pure_neumann = Sys2 singular). All runs use TRUE METIS geometric subdomains (1/rank).
set -u
cd "$(dirname "$0")/.."
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
geti(){ grep -oE "iters=[0-9]+" | head -1 | cut -d= -f2; }
CG="-ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12"
ASM="-pc_type asm -pc_asm_type basic -sub_ksp_type preonly -sub_pc_type icc"

echo "== A. anomaly + inexactness (Sys3 Laplace, nx=48, 4 METIS subdomains) =="
echo "method sub-solve O0 O1 O2 O3"
for s in 0 3; do for L in 0 1 2; do
  nm=$([ $s = 0 ] && echo BASIC || echo sASM); printf "%-6s ICC(%d) " $nm $L
  for ov in 0 1 2 3; do
    mpirun --allow-run-as-root -n 4 ./asm_demo -nx 48 -scheme $s $CG $ASM \
      -pc_asm_overlap $ov -sub_pc_factor_levels $L -ksp_max_it 1500 2>&1 | { printf "%4s " "$(geti)"; }
  done; echo
done; done
printf "%-6s LU     " BASIC
for ov in 0 1 2 3; do
  mpirun --allow-run-as-root -n 4 ./asm_demo -nx 48 -scheme 0 $CG -pc_type asm -pc_asm_type basic \
    -sub_ksp_type preonly -sub_pc_type lu -pc_asm_overlap $ov 2>&1 | { printf "%4s " "$(geti)"; }
done; echo

echo "== B. one-level variant bake-off (Sys3, nx=48, 4 sub, overlap=2, ICC0) =="
B="-nx 48 $CG -pc_asm_overlap 2 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0 -ksp_max_it 1500"
for cfg in "0 cg basic BASIC" "3 cg basic sASM" "4 cg basic sASM+Cheby" "7 cg basic epsPU" "8 cg basic SMRAS" "0 gmres restrict RAS"; do
  set -- $cfg
  v=$(mpirun --allow-run-as-root -n 4 ./asm_demo -scheme $1 -ksp_type $2 -pc_type asm -pc_asm_type $3 $B 2>&1 | geti)
  printf "  %-12s %s\n" "$4" "$v"
done

echo "== C. Sys2 (pure-Neumann singular = cardiac u_e), nx=48, 4 sub =="
for s in 0 3; do nm=$([ $s = 0 ] && echo BASIC || echo sASM); printf "%-6s " $nm
  for ov in 0 1 2; do
    mpirun --allow-run-as-root -n 4 ./asm_demo -nx 48 -pure_neumann -scheme $s $CG $ASM \
      -pc_asm_overlap $ov -sub_pc_factor_levels 0 -ksp_max_it 1500 2>&1 | { printf "%4s " "$(geti)"; }
  done; echo
done
printf "%-6s " GAMG; mpirun --allow-run-as-root -n 4 ./asm_demo -nx 48 -scheme 0 -ksp_type cg -pc_type gamg 2>&1 | geti

echo "== D. WEAK scaling (~15k dof/subdomain): one-level sASM vs two-level GAMG =="
echo "np nx dof sASM GAMG"
for cfg in "2 30" "4 38" "8 48" "16 60"; do
  set -- $cfg; np=$1; nx=$2; dof=$(( (nx+1)**3 ))
  s=$(mpirun --oversubscribe --allow-run-as-root -n $np ./asm_demo -nx $nx -scheme 3 $CG $ASM \
       -pc_asm_overlap 1 -sub_pc_factor_levels 0 -ksp_max_it 3000 2>&1 | geti)
  g=$(mpirun --oversubscribe --allow-run-as-root -n $np ./asm_demo -nx $nx -scheme 0 -ksp_type cg -pc_type gamg 2>&1 | geti)
  printf "%s %s %s %s %s\n" "$np" "$nx" "$dof" "${s:-X}" "${g:-X}"
done

echo "== E. sASM optimization space: local-solve accuracy & boundary transfer (Sys2) =="
B2="-nx 48 -pure_neumann -ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1500 -pc_type asm -pc_asm_type basic -sub_ksp_type preonly -sub_pc_type icc"
echo "method                     O=0 O=1 O=2 O=3   (Sys2 pure-Neumann, 4 METIS)"
row(){ printf "%-26s" "$1"; shift; for ov in 0 1 2 3; do
  mpirun --allow-run-as-root -n 4 ./asm_demo $B2 -pc_asm_overlap $ov "$@" 2>&1 | { printf "%4s" "$(geti)"; }; done; echo; }
row "BASIC ASM"               -scheme 0 -sub_pc_factor_levels 0
row "sASM ICC0 (overlap caps)" -scheme 3 -sub_pc_factor_levels 0
row "sASM ICC2"               -scheme 3 -sub_pc_factor_levels 2
row "sASM+Cheby4 (accurate)"  -scheme 4 -sub_pc_factor_levels 0 -localcheby 4
row "SMRAS (sym. RAS)"        -scheme 8 -sub_pc_factor_levels 0
row "eps-PU graded (dead end)" -scheme 7 -sub_pc_factor_levels 0
printf "%-26s" "GAMG two-level (ceiling)"; for ov in 0 1 2 3; do
  mpirun --allow-run-as-root -n 4 ./asm_demo -nx 48 -pure_neumann -scheme 0 -ksp_type cg -pc_type gamg 2>&1 | { printf "%4s" "$(geti)"; }; done; echo

echo "== F. large-scale cost: communication (VecScatter halo) & flops per method =="
BC2="-nx 48 -pure_neumann -ksp_type cg -ksp_norm_type preconditioned -ksp_rtol 1e-6 -ksp_atol 1e-12 -ksp_max_it 1500 -pc_type asm -pc_asm_type basic -pc_asm_overlap 1 -sub_ksp_type preonly -sub_pc_type icc -sub_pc_factor_levels 0"
costrow(){ local lbl="$1"; shift
  mpirun --allow-run-as-root -n 4 ./asm_demo $BC2 "$@" -log_view 2>&1 > /tmp/lv.txt
  it=$(grep -oE "iters=[0-9]+" /tmp/lv.txt|head -1|cut -d= -f2)
  vs=$(grep -E "^VecScatterBegin " /tmp/lv.txt|head -1|awk '{print $2}')
  tm=$(grep -oE "time=[0-9.]+" /tmp/lv.txt|head -1|cut -d= -f2)
  printf "%-14s iters=%-4s halo=%-6s halo/it=%-5s time=%s\n" "$lbl" "$it" "$vs" "$(awk "BEGIN{printf \"%.1f\",${vs:-0}/${it:-1}}")" "$tm"; }
costrow "BASIC"       -scheme 0
costrow "sASM"        -scheme 3
costrow "sASM+Cheby4" -scheme 4 -localcheby 4
costrow "SMRAS"       -scheme 8
