#!/bin/bash
# Noise/offset ensembles: soft convergence + distance to octahedral t48 reference, both crops.
set -u
cd /dev/shm/tta
for crop in crop512 crop512b; do
  for f in t0 t8 t48 n4 n8 o3 t8n4; do
    [ -f ${crop}_${f}.nrrd ] || /root/fenix export ${crop}_${f}.fxvol ${crop}_${f}.nrrd >/dev/null 2>&1
  done
  for pair in "n4 t48" "n8 t48" "o3 t48" "t8n4 t48" "n4 n8" "n8 t0" "o3 t0" "t8n4 t8"; do
    set -- $pair
    m=$(/root/fenix eval ${crop}_$1.fxvol ${crop}_$2.fxvol --thresh 0.5 --gt-thresh 0.5 2>&1 | grep official | awk "{print \$4}")
    s=$(/root/fenix compare ${crop}_$1.nrrd ${crop}_$2.nrrd 2>&1 | grep -oiE "MAE[= ]+[0-9.]+")
    echo "PAIR crop=$crop $1-vs-$2 mask=$m soft=$s"
  done
  rm -f ${crop}_*.nrrd
done
echo NZ_EVAL_DONE
