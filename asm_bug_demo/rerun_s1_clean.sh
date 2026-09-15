#!/bin/sh
# Re-run stride=1 with nothing else on the machine, so its wall time is
# comparable with the other strides (the first stride=1 run overlapped the
# parallel probe runs and its times are inflated by contention).
cd /home/user/cg-asm-icc-research/asm_bug_demo
until grep -q ALLDONE run_stride.log 2>/dev/null; do sleep 15; done
./asm_stride_test 24 6000 0.01 1 16 0 1x2x2 1 1 8 stride_probe_ref.csv \
    > stride_s1.csv 2> stride_s1.err
echo "S1CLEAN rc=$?" >> run_stride.log
