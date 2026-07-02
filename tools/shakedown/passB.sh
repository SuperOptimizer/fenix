#!/bin/sh
cd /root
F=/workspace/fenix/build-release/fenix
PY=/root/trainenv/bin/python

echo "=== B1 determinism: same seed => identical draws ==="
for R in 1 2; do
  rm -f /dev/shm/b$R.ring
  $F train-feed pairs4.txt /dev/shm/b$R.ring patch=128 slots=32 threads=8 octa=1 thickness=6 count=32 seed=777 > /dev/null 2>&1
  $PY /root/ring_dump.py /dev/shm/b$R.ring meta > det_$R.txt
done
if diff -q det_1.txt det_2.txt > /dev/null; then echo "B1_determinism=PASS ($(wc -l < det_1.txt) draws identical)"; else echo "B1_determinism=FAIL"; diff det_1.txt det_2.txt | head -4; fi

echo "=== B2 crash integrity: kill -9 mid-cold-fill, cache must reopen + refill ==="
rm -f /workspace/cache_crash.fxvol /dev/shm/c.ring
ZARR="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"
sed "s|/workspace/cache_p4.fxvol|/workspace/cache_crash.fxvol|" pairs4.txt > pairs_crash.txt
$F train-feed pairs_crash.txt /dev/shm/c.ring patch=128 slots=16 threads=8 octa=0 thickness=6 count=64 > feedc.log 2>&1 &
FP=$!
sleep 6
kill -9 $FP 2>/dev/null
echo "killed -9 mid-fill; cache: $(ls -la /workspace/cache_crash.fxvol | awk '{print $5}') bytes"
# reopen + continue: a fresh feed run against the same crashed cache must succeed
rm -f /dev/shm/c.ring
$F train-feed pairs_crash.txt /dev/shm/c.ring patch=128 slots=16 threads=8 octa=0 thickness=6 count=16 > feedc2.log 2>&1
RC=$?
grep -o "worker-time.*" feedc2.log
[ $RC -eq 0 ] && echo "B2_crash_integrity=PASS" || { echo "B2_crash_integrity=FAIL"; tail -n 3 feedc2.log; }

echo "=== B3 label content sanity ==="
rm -f /dev/shm/s.ring
$F train-feed pairs4.txt /dev/shm/s.ring patch=128 slots=16 threads=8 octa=1 thickness=6 count=16 seed=99 > /dev/null 2>&1
$PY /root/ring_dump.py /dev/shm/s.ring content > content.txt
cat content.txt
$PY - <<'PYEOF'
rows = [eval(l) for l in open('/root/content.txt')]
bad = [r for r in rows if r[2] == 0]           # no sheet voxels at all (sampler centers ON surface)
flat = [r for r in rows if r[1] < 1.0]          # CT basically uniform = suspicious gather
print(f"B3: {len(rows)} slots, {len(bad)} without sheet labels, {len(flat)} with flat CT")
print("B3_labels=" + ("PASS" if rows and not bad and not flat else "FAIL"))
PYEOF
