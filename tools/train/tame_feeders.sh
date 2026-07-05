#!/bin/sh
# stop ALL feed_loops + feeders, then start exactly ONE per ring (cache is warm; the
# training loop reads the ring, feeders just top it up). Does NOT touch train.py.
pkill -f "feed_loop" 2>/dev/null
pkill -f inst_sup.sh 2>/dev/null
pkill -x fenix 2>/dev/null
sleep 4
F=/workspace/fenix/build-release/fenix
nohup $F train-feed /root/pairs_inst_full.txt /dev/shm/inst.ring patch=128 slots=96 threads=12 aug=2 locality=16 cache_mb=6144 thickness=6 wrapk=8 >> /root/inst_feed.log 2>&1 &
nohup $F train-feed /root/pairs_inst_val.txt /dev/shm/instval.ring patch=128 slots=32 threads=2 aug=0 locality=1 cache_mb=1024 thickness=6 wrapk=8 >> /root/instval_feed.log 2>&1 &
echo "tamed: 1 feeder per ring"
