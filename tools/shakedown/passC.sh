#!/bin/sh
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python

echo "=== C1 multi-volume corpus (P4 on-demand + 1447 local crop) ==="
cat pairs4.txt > pairs_multi.txt
printf '/root/seg1447.fxsurf /root/ct1447.fxvol - 12000 2300 3200\n' >> pairs_multi.txt
rm -f /dev/shm/m.ring
$F train-feed pairs_multi.txt /dev/shm/m.ring patch=128 slots=24 threads=8 octa=1 thickness=6 count=100 seed=5 > feedm.log 2>&1 &
FP=$!
$PY /root/drain.py /dev/shm/m.ring 300 > drainm.log 2>&1 &
DP=$!
wait $FP; RC=$?
kill $DP 2>/dev/null
grep -o "worker-time.*" feedm.log
[ $RC -eq 0 ] && echo "C1_multivolume=PASS" || { echo "C1_multivolume=FAIL"; tail -n 3 feedm.log; }

echo "=== C2 1000-step soak: RSS + throughput drift ==="
rm -f /dev/shm/soak.ring
$F train-feed pairs_multi.txt /dev/shm/soak.ring patch=128 slots=32 threads=12 octa=1 thickness=6 > feedsoak.log 2>&1 &
FP=$!
sleep 5
( while true; do
    FR=$(ps -o rss= -p $FP 2>/dev/null); TR=$(ps -o rss= -p $(pgrep -f "steps 1000" | head -1) 2>/dev/null)
    [ -z "$FR" ] && break
    echo "$(date +%s) feeder_rss_mb=$((FR/1024)) train_rss_mb=$((${TR:-0}/1024))"
    sleep 20
  done > rss.log 2>&1 ) &
MON=$!
$PY /workspace/fenix/tools/train/train.py --ring /dev/shm/soak.ring --steps 1000 --batch 8 --base 16 --lr 5e-4 --out /root/s_soak --ckpt-every 500 > soak.log 2>&1
RC=$?
kill $FP $MON 2>/dev/null
echo "--- soak stats (first/mid/last):"
grep "^step" soak.log | sed -n '1p;10p;20p'
echo "--- RSS trajectory (first/last):"
head -1 rss.log; tail -1 rss.log
[ $RC -eq 0 ] && echo "C2_soak=PASS" || { echo "C2_soak=FAIL"; tail -n 4 soak.log; }
