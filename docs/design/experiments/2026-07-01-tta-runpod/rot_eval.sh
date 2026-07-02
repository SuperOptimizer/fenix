#!/bin/bash
# Rotation-family vs octahedral-family: mask agreement + soft MAE/PSNR.
set -u
cd /dev/shm/tta
for crop in crop512 crop512b; do
  for f in t0 t48 t48ms r4 r8 t8r4; do
    [ -f ${crop}_${f}.nrrd ] || /root/fenix export ${crop}_${f}.fxvol ${crop}_${f}.nrrd > /dev/null 2>&1
  done
  for pair in "r4 r8" "r8 t48" "r4 t48" "t8r4 t48" "r8 t48ms" "t8r4 t48ms" "r8 t0"; do
    set -- $pair
    m=$(/root/fenix eval ${crop}_$1.fxvol ${crop}_$2.fxvol --thresh 0.5 --gt-thresh 0.5 2>&1 | grep official | awk "{print \$4}")
    s=$(/root/fenix compare ${crop}_$1.nrrd ${crop}_$2.nrrd 2>&1 | grep -iE "psnr" )
    echo "PAIR crop=$crop $1-vs-$2 mask=$m soft: $s"
  done
  rm -f ${crop}_*.nrrd
done
echo ROT_EVAL_DONE
