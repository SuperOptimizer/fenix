#!/bin/sh
# Round D: trainer balance sweep + multi-resolution canonical feed + validation-ring cadence
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python
TT=/workspace/fenix/tools/train

echo "===== D1 trainer balance: base x batch (patch=128, 60 steps each, warm feed) ====="
rm -f /dev/shm/d1.ring
$F train-feed pairs4.txt /dev/shm/d1.ring patch=128 slots=48 threads=16 octa=1 thickness=6 cache_mb=8192 > feedd1.log 2>&1 &
FP=$!
sleep 8
for BASE in 16 32 48; do
  for B in 4 8 12; do
    OUT=$($PY $TT/train.py --ring /dev/shm/d1.ring --steps 60 --batch $B --base $BASE --out /root/s_d1 --ckpt-every 9999 2>&1 | grep "^step 50" | head -1)
    echo "base=$BASE batch=$B: $OUT"
  done
done
kill $FP 2>/dev/null
echo "D1 done (pick the config with best patches/s at feedwait<50%)"

echo "===== D2 multi-resolution canonical feed (2.4um P4 + 8.64um 1447 resampled) ====="
cat pairs4.txt > pairs_mr.txt
printf '/root/seg1447.fxsurf /root/ct1447.fxvol - 12000 2300 3200 um=8.64\n' >> pairs_mr.txt
rm -f /dev/shm/d2.ring
$F train-feed pairs_mr.txt /dev/shm/d2.ring patch=128 slots=24 threads=8 octa=1 thickness=6 cache_mb=8192 count=100 seed=11 > feedd2.log 2>&1 &
FP=$!
$PY /root/drain.py /dev/shm/d2.ring 240 > draind2.log 2>&1 &
DP=$!
wait $FP; RC=$?
kill $DP 2>/dev/null
grep -o "worker-time.*" feedd2.log
[ $RC -eq 0 ] && echo "D2_multires=PASS" || { echo "D2_multires=FAIL"; tail -n 3 feedd2.log; }

echo "===== D3 validation-ring cadence (train P4 meshes 1-4, validate on mesh 5) ====="
head -4 pairs4.txt > pairs_tr.txt
tail -1 pairs4.txt > pairs_va.txt
rm -f /dev/shm/d3t.ring /dev/shm/d3v.ring
$F train-feed pairs_tr.txt /dev/shm/d3t.ring patch=128 slots=32 threads=12 octa=1 thickness=6 cache_mb=8192 > feedd3t.log 2>&1 &
FT=$!
$F train-feed pairs_va.txt /dev/shm/d3v.ring patch=128 slots=16 threads=4 octa=0 thickness=6 cache_mb=4096 seed=999 > feedd3v.log 2>&1 &
FV=$!
sleep 8
$PY $TT/train.py --ring /dev/shm/d3t.ring --val-ring /dev/shm/d3v.ring --val-every 100 --steps 300 \
  --batch 8 --base 16 --out /root/s_d3 --ckpt-every 9999 > d3.log 2>&1
RC=$?
kill $FT $FV 2>/dev/null
grep -E "VAL" d3.log
[ $RC -eq 0 ] && grep -q "VAL" d3.log && echo "D3_valring=PASS" || { echo "D3_valring=FAIL"; tail -n 4 d3.log; }
