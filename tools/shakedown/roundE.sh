#!/bin/sh
# Round E — dress rehearsal: KD-train a small student -> export .ts -> fenix predict-surface
# with it -> official eval vs rasterized GT (teacher scored as reference). PASS = mechanics
# (every stage runs, numbers produced); quality comes from real training later.
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python
TT=/workspace/fenix/tools/train

echo "===== E1 KD training (1447 crop + teacher, 400 steps) ====="
printf '/root/seg1447.fxsurf /root/ct1447.fxvol /root/teacher1447.fxvol 12000 2300 3200 um=8.64\n' > pairs_e.txt
rm -f /dev/shm/e.ring
$F train-feed pairs_e.txt /dev/shm/e.ring patch=128 slots=24 threads=8 octa=1 thickness=3 cache_mb=4096 > feede.log 2>&1 &
FP=$!
sleep 5
$PY $TT/train.py --ring /dev/shm/e.ring --steps 400 --batch 6 --base 16 --lr 1e-3 \
  --alpha 0.7 --beta 0.3 --out /root/s_e --ckpt-every 9999 > e_train.log 2>&1
RC=$?; kill $FP 2>/dev/null
tail -n 2 e_train.log
[ $RC -eq 0 ] && [ -f /root/s_e_final.ts ] && echo "E1_train_export=PASS" || { echo "E1_train_export=FAIL"; tail -n 5 e_train.log; exit 1; }

echo "===== E2 student inference via .ts ====="
rm -f /root/pred_student.fxvol
$F predict-surface /root/ct1447.fxvol /root/s_e_final.ts /root/pred_student.fxvol 128 0.5 1 > e_pred.log 2>&1
RC=$?
tail -n 1 e_pred.log
[ $RC -eq 0 ] && echo "E2_ts_inference=PASS" || { echo "E2_ts_inference=FAIL"; tail -n 4 e_pred.log; exit 1; }

echo "===== E3 GT + eval firewall ====="
# GT rasterized from the mesh over the crop's ABSOLUTE box (same voxels as the crop, NOTE: at
# source 8.64um resolution — student predicted on the source-res crop, so grids align)
if [ ! -f /root/gt1447.fxvol ]; then
  $F surfaces 12000 2300 3200 768 768 768 /root/seg1447.fxsurf out=/root/gt1447.fxvol thickness=3 shell=0 > e_gt.log 2>&1 || { echo "E3 GT rasterize FAIL"; tail -n 3 e_gt.log; exit 1; }
fi
echo "--- student vs GT:"
$F eval /root/pred_student.fxvol /root/gt1447.fxvol 2>&1 | tail -n 6
echo "--- teacher vs GT (reference):"
$F eval /root/teacher1447.fxvol /root/gt1447.fxvol 2>&1 | tail -n 6
echo "ROUND_E_MECHANICS=PASS"
