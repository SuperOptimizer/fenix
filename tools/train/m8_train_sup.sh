#!/bin/sh
# M8 trainer supervisor: if a trainer dies (e.g. feed gap > timeout), restart resuming
# from its newest step checkpoint. Exits when the final checkpoint exists.
arm=$1; gpu=$2
ring=/dev/shm/feed$arm.ring
out=/tmp/gtqc/m8/student$arm
log=/tmp/gtqc/m8/train$arm.log
N=0
while [ ! -f ${out}_final.pt ] && [ $N -lt 50 ]; do
  if pgrep -f "train.py --ring $ring" >/dev/null 2>&1; then sleep 60; continue; fi
  ck=$(ls -t ${out}_step*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] && R="--resume $ck" || R=""
  CUDA_VISIBLE_DEVICES=$gpu python3 /home/forrest/fenix/tools/train/train.py \
    --ring $ring --steps 12000 --batch 8 --beta 1.0 --feed-timeout 1800 \
    --out $out --ckpt-every 2000 $R >> $log 2>&1
  echo "$(date +%T) SUP: train$arm exited (restart $N, resume='$ck')" >> /tmp/gtqc/m8/sup.log
  N=$((N+1)); sleep 20
done
echo "$(date +%T) SUP: train$arm DONE" >> /tmp/gtqc/m8/sup.log
