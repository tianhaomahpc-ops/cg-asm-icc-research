#!/bin/sh
# Second batch: at a 1 ms solve interval the snapshots are 100x further apart,
# so the right window size need not be the one measured at 0.01 ms.  Each run
# reports m and 4m, so win = 4 / 8 covers m = 4, 8, 16, 32.
cd /home/user/cg-asm-icc-research/asm_bug_demo
N=24; NT=6000; DT=0.01; REG=0; SUB=1x2x2; OVL=1; PC=1; TOL=8
for W in 4 8; do
    ./asm_stride_test $N $NT $DT 100 $W $REG $SUB $OVL $PC $TOL \
        > stride_s100_w${W}.csv 2> stride_s100_w${W}.err
    echo "done stride=100 win=$W rc=$?"
done
# and the same window sweep at stride=1, for the side-by-side
for W in 4 8; do
    ./asm_stride_test $N $NT $DT 1 $W $REG $SUB $OVL $PC $TOL \
        > stride_s1_w${W}.csv 2> stride_s1_w${W}.err
    echo "done stride=1 win=$W rc=$?"
done
echo BATCH2DONE
