#!/bin/sh
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python
echo "=== pass A: feeder throughput sweep (warm cache, patch=128, count=300) ==="
for T in 4 8 16 24; do
  rm -f /dev/shm/a.ring
  $PY /root/drain.py /dev/shm/a.ring 240 > drain_$T.log 2>&1 &
  DP=$!
  sleep 1
  S=$(date +%s.%N)
  $F train-feed pairs4.txt /dev/shm/a.ring patch=128 slots=32 threads=$T octa=1 thickness=6 count=300 > feedA_$T.log 2>&1
  E=$(date +%s.%N)
  kill $DP 2>/dev/null; wait $DP 2>/dev/null
  W=$(echo "$E $S" | awk '{printf "%.1f", $1-$2}')
  echo "threads=$T wall=${W}s  $(grep -o 'worker-time.*' feedA_$T.log)  => $(echo "$W" | awk '{printf "%.1f", 300/$1}') draws/s"
done
