#!/bin/sh
# Shakeout suite — each phase prints PHASE_<name>=PASS/FAIL and key numbers.
cd /root
F=/workspace/fenix/build-release/fenix
TT=/workspace/fenix/tools/train
PY=/root/trainenv/bin/python
ZARR="https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/0"

phase() { echo; echo "===== PHASE $1 ====="; }

feed() { # $1 ring  $2 pairs  $3 extra
  $F train-feed "$2" "$1" patch=128 slots=24 threads=8 octa=1 thickness=6 $3 > /root/feed_cur.log 2>&1 &
  echo $!
}

phase warm_cache_speed
rm -f /dev/shm/w.ring
FP=$(feed /dev/shm/w.ring pairs4.txt)
sleep 5
$PY $TT/train.py --ring /dev/shm/w.ring --steps 150 --batch 8 --base 16 --lr 1e-3 --out /root/s_warm --ckpt-every 1000 > /root/warm.log 2>&1
RC=$?; kill $FP 2>/dev/null
tail -n 2 /root/warm.log
FW=$(grep -o "feedwait [0-9]*%" /root/warm.log | tail -1)
[ $RC -eq 0 ] && echo "PHASE_warm=PASS ($FW)" || { echo "PHASE_warm=FAIL"; tail -n 5 /root/warm.log; }

phase resume_training
rm -f /dev/shm/r.ring /root/s_res*
FP=$(feed /dev/shm/r.ring pairs4.txt)
sleep 5
$PY $TT/train.py --ring /dev/shm/r.ring --steps 120 --batch 8 --base 16 --out /root/s_res --ckpt-every 25 > /root/res1.log 2>&1 &
TP=$!
sleep 75 && kill $TP 2>/dev/null; wait $TP 2>/dev/null
CK=$(ls -t /root/s_res_step*.pt 2>/dev/null | head -1)
echo "killed training; ckpt: $CK"
if [ -n "$CK" ]; then
  $PY $TT/train.py --ring /dev/shm/r.ring --steps 120 --batch 8 --base 16 --out /root/s_res --ckpt-every 25 --resume "$CK" > /root/res2.log 2>&1
  RC=$?
  grep -m1 "resumed" /root/res2.log
  [ $RC -eq 0 ] && grep -q "resumed" /root/res2.log && echo "PHASE_resume=PASS" || { echo "PHASE_resume=FAIL"; tail -n 4 /root/res2.log; }
else
  echo "PHASE_resume=FAIL (no checkpoint written before kill)"
fi
kill $FP 2>/dev/null

phase feeder_death_recovery
rm -f /dev/shm/d.ring
FP=$(feed /dev/shm/d.ring pairs4.txt)
sleep 5
$PY $TT/train.py --ring /dev/shm/d.ring --steps 100 --batch 8 --base 16 --out /root/s_dead --ckpt-every 1000 > /root/dead.log 2>&1 &
TP=$!
sleep 20
kill $FP 2>/dev/null; echo "feeder killed at t=20s"
sleep 10
FP=$(feed /dev/shm/d.ring pairs4.txt)   # restart feeder onto the SAME ring
echo "feeder restarted"
wait $TP; RC=$?
kill $FP 2>/dev/null
tail -n 2 /root/dead.log
[ $RC -eq 0 ] && echo "PHASE_feeder_death=PASS" || { echo "PHASE_feeder_death=FAIL"; tail -n 4 /root/dead.log; }

phase teacher_channel_kd
if [ ! -f /root/teacher1447.fxvol ]; then
  $F predict-surface /root/ct1447.fxvol /root/surface.fxweights /root/teacher1447.fxvol 2>&1 | tail -1
fi
printf '/root/seg1447.fxsurf /root/ct1447.fxvol /root/teacher1447.fxvol 12000 2300 3200\n' > pairs_kd.txt
rm -f /dev/shm/k.ring
FP=$($F train-feed pairs_kd.txt /dev/shm/k.ring patch=128 slots=16 threads=6 octa=1 thickness=3 > /root/feedk.log 2>&1 & echo $!)
sleep 5
$PY $TT/train.py --ring /dev/shm/k.ring --steps 80 --batch 4 --base 16 --alpha 0.7 --beta 0.3 --out /root/s_kd --ckpt-every 1000 > /root/kd.log 2>&1
RC=$?; kill $FP 2>/dev/null
KD=$(grep -o "kd [0-9.]*" /root/kd.log | tail -1)
tail -n 2 /root/kd.log
[ $RC -eq 0 ] && [ -n "$KD" ] && echo "PHASE_kd=PASS (last $KD)" || { echo "PHASE_kd=FAIL"; tail -n 4 /root/kd.log /root/feedk.log; }

phase patch256_profile_batch
rm -f /dev/shm/p.ring
FP=$($F train-feed pairs4.txt /dev/shm/p.ring patch=256 slots=10 threads=8 octa=1 thickness=6 > /root/feedp.log 2>&1 & echo $!)
sleep 8
$PY $TT/train.py --ring /dev/shm/p.ring --steps 40 --batch 2 --base 16 --out /root/s_p256 --ckpt-every 1000 > /root/p256.log 2>&1
RC=$?; kill $FP 2>/dev/null
tail -n 2 /root/p256.log
[ $RC -eq 0 ] && echo "PHASE_p256=PASS" || { echo "PHASE_p256=FAIL"; tail -n 4 /root/p256.log /root/feedp.log; }

phase qat_flag
/root/trainenv/bin/pip install -q torchao 2>&1 | tail -1
rm -f /dev/shm/q.ring
FP=$($F train-feed pairs4.txt /dev/shm/q.ring patch=128 slots=16 threads=8 octa=1 thickness=6 > /root/feedq.log 2>&1 & echo $!)
sleep 5
$PY $TT/train.py --ring /dev/shm/q.ring --steps 30 --batch 4 --base 16 --qat --out /root/s_qat --ckpt-every 1000 > /root/qat.log 2>&1
RC=$?; kill $FP 2>/dev/null
grep -m1 -i "qat" /root/qat.log
[ $RC -eq 0 ] && echo "PHASE_qat=PASS" || { echo "PHASE_qat=FAIL"; tail -n 5 /root/qat.log; }

echo; echo "===== SUITE DONE ====="
grep -h "PHASE_" /dev/null 2>/dev/null
