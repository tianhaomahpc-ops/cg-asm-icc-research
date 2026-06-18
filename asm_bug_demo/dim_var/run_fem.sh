#!/bin/bash
# run_fem.sh -- full unstructured-P1-FEM ASM-vs-sASM study (cases a,b,c,d).
# Builds meshes (fem_build.py) then sweeps overlap for BOX and METIS partitions,
# BASIC (scheme 0) and sASM (scheme 3); emits [RESULT] to results/fem.log.
set -u
cd "$(dirname "$0")"
mkdir -p results mesh
LOG=results/fem.log; : > "$LOG"
OVS="0 1 2 3 4"

# (case,dim,n_per_axis,nsub)  -- aniso (b,d) degenerate in 1D -> skip 1D for b,d
SPECS="
a 1 4000 8
a 2 70 16
a 3 20 27
b 2 70 16
b 3 20 27
c 1 4000 8
c 2 70 16
c 3 20 27
d 2 70 16
d 3 20 27
"

echo "### build meshes" | tee -a "$LOG"
echo "$SPECS" | while read cs dm n ns; do
  [ -z "$cs" ] && continue
  python3 fem_build.py "$cs" "$dm" "$n" "$ns" 2>&1 | tee -a "$LOG"
done

echo "### solver sweeps" | tee -a "$LOG"
echo "$SPECS" | while read cs dm n ns; do
  [ -z "$cs" ] && continue
  tagbase="case${cs}_${dm}D"
  K="mesh/${tagbase}_K.petsc"; B="mesh/${tagbase}_b.petsc"
  for part in box metis; do
    P="mesh/${tagbase}_${part}.is"
    for sc in 0 3; do
      for ov in $OVS; do
        eig=""
        # dump Ritz spectrum for a representative cell (2D, O=2)
        if [ "$dm" = 2 ] && [ "$ov" = 2 ] && [ "$part" = box ]; then
          eig="-eig_out results/eig_${tagbase}_${part}_s${sc}_O${ov}.txt"
        fi
        ./schwarz_fem -K "$K" -b "$B" -part "$P" -scheme $sc -overlap $ov \
            -icc_levels 0 -measure_omega -tag "${tagbase}_${part}" $eig \
            -sub_pc_factor_shift_type positive_definite 2>/dev/null \
            | grep '\[RESULT\]' | tee -a "$LOG"
      done
    done
  done
done
echo "wrote $LOG"
