#!/bin/bash
# How much closer to the best-estimate reference (t48ms) is full TTA vs a single raw pass?
# Reports soft MAE + PSNR and mask official, both crops. Lower MAE / higher PSNR & official = closer.
set -u
cd /dev/shm/tta
for crop in crop512 crop512b; do
  ref=${crop}_t48ms
  [ -f ${ref}.nrrd ] || /root/fenix export ${ref}.fxvol ${ref}.nrrd >/dev/null 2>&1
  for combo in t0 t8 t24 t48; do
    [ -f ${crop}_${combo}.nrrd ] || /root/fenix export ${crop}_${combo}.fxvol ${crop}_${combo}.nrrd >/dev/null 2>&1
    soft=$(/root/fenix compare ${crop}_${combo}.nrrd ${ref}.nrrd 2>&1 | grep -oiE "MAE[= ]+[0-9.]+|PSNR[= ]+[0-9.]+")
    mask=$(/root/fenix eval ${crop}_${combo}.fxvol ${ref}.fxvol --thresh 0.5 --gt-thresh 0.5 2>&1 | grep official | awk "{print \$4}")
    echo "GAIN crop=$crop combo=$combo mask_official=$mask $(echo $soft | tr '\n' ' ')"
  done
  rm -f ${crop}_*.nrrd
done
echo GAIN_DONE
