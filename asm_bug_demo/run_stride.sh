#!/bin/sh
# Sys2/Sys3 solve frequency sweep.  Runs SEQUENTIALLY: iteration counts are
# deterministic but wall times are not, and two concurrent runs inflate each
# other by ~10%.
cd /home/user/cg-asm-icc-research/asm_bug_demo
N=24; NT=6000; DT=0.01; WIN=16; REG=0; SUB=1x2x2; OVL=1; PC=1; TOL=8
for S in 1 10 25 50 100 200; do
    if [ "$S" = 1 ]; then PROBE=stride_probe_ref.csv; else PROBE=""; fi
    ./asm_stride_test $N $NT $DT $S $WIN $REG $SUB $OVL $PC $TOL $PROBE \
        > stride_s${S}.csv 2> stride_s${S}.err
    echo "done stride=$S rc=$?"
done
echo ALLDONE
