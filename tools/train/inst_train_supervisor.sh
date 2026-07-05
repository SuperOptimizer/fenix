#!/bin/sh
L=/root/inst_sup.log
echo "$(date +%T) SUP v2 start" > $L
pkill -f inst_run.sh 2>/dev/null; pkill -x fenix 2>/dev/null; pkill -x python3 2>/dev/null
sleep 3
F=/workspace/fenix/build-release/fenix
feed_loop() {  # $1 pairs $2 ring $3 log $4 extra-args...
  P=$1; R=$2; LG=$3; shift 3
  N=0
  while [ $N -lt 60 ]; do
    $F train-feed "$P" "$R" patch=128 thickness=6 wrapk=8 "$@" >> "$LG" 2>&1
    echo "$(date +%T) SUP feeder $R exited (restart $N)" >> /root/inst_sup.log
    N=$((N+1)); sleep 15
  done
}
feed_loop /root/pairs_inst_full.txt /dev/shm/inst.ring /root/inst_feed.log slots=96 threads=12 aug=2 locality=16 cache_mb=6144 &
feed_loop /root/pairs_inst_val.txt /dev/shm/instval.ring /root/instval_feed.log slots=32 threads=2 aug=0 locality=1 cache_mb=1024 &
echo "$(date +%T) SUP feeders launched (self-restarting)" >> $L
N=0
while [ ! -f /dev/shm/inst.ring ] || [ ! -f /dev/shm/instval.ring ]; do sleep 5; N=$((N+5)); [ $N -gt 900 ] && { echo "$(date +%T) SUP RING TIMEOUT" >> $L; exit 1; }; done
echo "$(date +%T) SUP rings ready" >> $L
sleep 30
echo "$(date +%T) SUP starting train.py" >> $L
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 /root/train.py --ring /dev/shm/inst.ring --val-ring /dev/shm/instval.ring --val-every 500 \
  --steps 25000 --batch 4 --base 32 --alpha 0 --cldice 0 --lr 3e-4 --wrapk 8 --feed-timeout 900 \
  --out /workspace/run/inst_w8g --ckpt-every 2000 > /root/inst_train.log 2>&1
echo "$(date +%T) SUP train exit $?" >> $L
