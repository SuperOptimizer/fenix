#!/bin/sh
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python
TT=/workspace/fenix/tools/train

echo "===== G: full-corpus feed ====="
python3 /root/gen_pairs.py
S=$(date +%s.%N)
rm -f /dev/shm/g.ring
$PY /root/drain.py /dev/shm/g.ring 600 > draing.log 2>&1 &
DP=$!
/usr/bin/time -v $F train-feed pairs_full.txt /dev/shm/g.ring patch=128 slots=48 threads=16 octa=1 thickness=6 cache_mb=2048 count=500 seed=21 > feedg.log 2>&1
RC=$?
E=$(date +%s.%N)
kill $DP 2>/dev/null
grep -o "worker-time.*" feedg.log
grep -E "Maximum resident|Elapsed" feedg.log | head -2
echo "G wall=$(echo "$E $S" | awk '{printf "%.0f", $1-$2}')s rc=$RC ($(echo "$E $S" | awk '{printf "%.1f", 500/($1-$2)}') draws/s incl. startup)"
[ $RC -eq 0 ] && echo "G_fullcorpus=PASS" || { echo "G_fullcorpus=FAIL"; tail -n 3 feedg.log; }

echo "===== H1: warm scaling sweep with u8 fast path ====="
for T in 16 24 32 48; do
  rm -f /dev/shm/h.ring
  $PY /root/drain.py /dev/shm/h.ring 240 > drainh.log 2>&1 &
  DP=$!
  S=$(date +%s.%N)
  $F train-feed pairs4.txt /dev/shm/h.ring patch=128 slots=64 threads=$T octa=1 thickness=6 cache_mb=8192 count=400 > feedh_$T.log 2>&1
  E=$(date +%s.%N)
  kill $DP 2>/dev/null
  echo "threads=$T $(echo "$E $S" | awk '{printf "%.1f", 400/($1-$2)}') draws/s  $(grep -o 'worker-time.*' feedh_$T.log)"
done

echo "===== H2: two concurrent feeders (multi-GPU data-plane rehearsal) ====="
rm -f /dev/shm/h2a.ring /dev/shm/h2b.ring
$PY /root/drain.py /dev/shm/h2a.ring 240 > drainh2a.log 2>&1 &
DA=$!
$PY /root/drain.py /dev/shm/h2b.ring 240 > drainh2b.log 2>&1 &
DB=$!
S=$(date +%s.%N)
$F train-feed pairs4.txt /dev/shm/h2a.ring patch=128 slots=48 threads=16 octa=1 thickness=6 cache_mb=8192 count=300 > feedh2a.log 2>&1 &
FA=$!
$F train-feed pairs_va.txt /dev/shm/h2b.ring patch=128 slots=48 threads=16 octa=1 thickness=6 cache_mb=8192 count=300 seed=999 > feedh2b.log 2>&1 &
FB=$!
wait $FA; RA=$?
wait $FB; RB=$?
E=$(date +%s.%N)
kill $DA $DB 2>/dev/null
echo "2 feeders x300 draws in $(echo "$E $S" | awk '{printf "%.0f", $1-$2}')s => aggregate $(echo "$E $S" | awk '{printf "%.1f", 600/($1-$2)}') draws/s"
[ $RA -eq 0 ] && [ $RB -eq 0 ] && echo "H2_dual_feeder=PASS" || echo "H2_dual_feeder=FAIL"

echo "===== I: GPU-side training matrix (warm feed, 150 steps each, base=32 batch=8) ====="
rm -f /dev/shm/i.ring
$F train-feed pairs4.txt /dev/shm/i.ring patch=128 slots=64 threads=24 octa=1 thickness=6 cache_mb=8192 > feedi.log 2>&1 &
FP=$!
sleep 8
for CFG in "baseline:" "pinned:--pinned" "fused:--fused-adam" "pinned+fused:--pinned --fused-adam" "compile:--compile --pinned --fused-adam"; do
  NAME=${CFG%%:*}; FLAGS=${CFG#*:}
  OUT=$($PY $TT/train.py --ring /dev/shm/i.ring --steps 150 --batch 8 --base 32 --prof $FLAGS --out /root/s_i --ckpt-every 9999 2>&1 | grep -E "^prof|^step 100" | tail -n 2)
  echo "[$NAME] $OUT"
done
kill $FP 2>/dev/null
echo "ROUND_GHI_DONE"
