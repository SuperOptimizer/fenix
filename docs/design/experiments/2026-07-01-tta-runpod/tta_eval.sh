#!/bin/bash
# Score each combo against the maximal ensemble (t48ms) per crop, at auto and fixed-0.5
# thresholds, + convergence pairs. fenix logs to stderr -> 2>&1 everywhere.
set -u
cd /dev/shm/tta
for crop in crop512 crop512b; do
  ref=${crop}_t48ms.fxvol
  for combo in t0 t8 t24 t48 s12 ms t8ms; do
    a=$(/root/fenix eval ${crop}_${combo}.fxvol $ref 2>&1 | grep "official" | awk "{print \$4}")
    f=$(/root/fenix eval ${crop}_${combo}.fxvol $ref --thresh 0.5 --gt-thresh 0.5 2>&1 | grep "official" | awk "{print \$4}")
    echo "SCORE crop=$crop combo=$combo vs_ref_auto=$a vs_ref_fix05=$f"
  done
  for pair in "t24 t48" "t48 t48ms" "t8ms t48ms"; do
    set -- $pair
    f=$(/root/fenix eval ${crop}_$1.fxvol ${crop}_$2.fxvol --thresh 0.5 --gt-thresh 0.5 2>&1 | grep "official" | awk "{print \$4}")
    echo "CONV crop=$crop $1-vs-$2 fix05=$f"
  done
done
echo EVAL_DONE
