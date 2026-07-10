#!/bin/sh
# M8 feeder supervisor: restart-on-exit (WAN hiccups kill a feeder ~hourly; the ring file
# survives — same inode — so the trainer's mmap stays valid and feed-timeout rides the gap).
F=/home/forrest/fenix/build-release/fenix
export FENIX_ZARR_FETCH_THREADS=8
feed_loop() { # $1 pairs $2 ring $3 log
  P=$1; R=$2; LG=$3
  N=0
  while [ $N -lt 100 ]; do
    $F train-feed "$P" "$R" patch=128 slots=32 threads=12 seed=42 disk_mb=131072 >> "$LG" 2>&1
    echo "$(date +%T) SUP: feeder $R exited (restart $N)" >> /tmp/gtqc/m8/sup.log
    N=$((N+1)); sleep 10
  done
}
: > /tmp/gtqc/m8/sup.log
feed_loop /tmp/gtqc/pairs_graded2.txt /dev/shm/feedG.ring /tmp/gtqc/m8/feedG.log &
feed_loop /tmp/gtqc/pairs_raw2.txt /dev/shm/feedR.ring /tmp/gtqc/m8/feedR.log &
wait
