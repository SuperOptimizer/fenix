#!/bin/sh
# One-command supervised training run: WAN-resilient feeder supervisors (train + val
# rings), trainer supervisor with checkpoint resume, and AUTO-EVAL when the final
# checkpoint lands (eval_students.py: dice/sd2 mean+p10+min and ECE on the standard
# holdout crops — both the final and best-val checkpoints are scored).
#
# Usage: train_run.sh <pairs.txt> <val_pairs.txt> <out-prefix> [args...]
#   args: BASE=32 STEPS=60000 BATCH=4 ACCUM=2 GPU=0 RING=/dev/shm/feedM.ring
#         VRING=/dev/shm/feedV.ring CROPS=/tmp/gtqc/m8/eval
# Lessons baked in: feeders die ~hourly on WAN (supervisor keeps the ring inode so the
# trainer's mmap survives); batch 8 base 32 @128^3 OOMs 16GB (use BATCH=4 ACCUM=2);
# training caches live on real disk, never tmpfs (/tmp quota turns reads into air).
set -u
PAIRS=$1; VPAIRS=$2; OUT=$3; shift 3
BASE=32; STEPS=60000; BATCH=4; ACCUM=2; GPU=0
RING=/dev/shm/feedM.ring; VRING=/dev/shm/feedV.ring; CROPS=/tmp/gtqc/m8/eval
for a in "$@"; do eval "$a"; done
D=$(dirname $OUT); mkdir -p $D
FENIX=${FENIX_BIN:-$(dirname "$0")/../../build-release/fenix}
export FENIX_ZARR_FETCH_THREADS=8

feed_loop() {
  P=$1; R=$2; LG=$3; shift 3
  N=0
  while [ $N -lt 200 ] && [ ! -f ${OUT}_final.pt ]; do
    $FENIX train-feed "$P" "$R" "$@" >> "$LG" 2>&1
    echo "$(date +%T) SUP: feeder $R exited (restart $N)" >> $D/sup.log
    N=$((N+1)); sleep 15
  done
}
feed_loop "$PAIRS" "$RING" $D/feedM.log \
  patch=128 slots=32 threads=12 seed=42 aug=2 disk_mb=131072 &
feed_loop "$VPAIRS" "$VRING" $D/feedV.log \
  patch=128 slots=8 threads=4 seed=777 aug=0 disk_mb=131072 &

N=0
while [ ! -f ${OUT}_final.pt ] && [ $N -lt 80 ]; do
  if pgrep -f "train.py --ring $RING" >/dev/null 2>&1; then sleep 60; continue; fi
  ck=$(ls -t ${OUT}_step*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] && R="--resume $ck" || R=""
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 $(dirname "$0")/train.py \
    --ring $RING --steps $STEPS --batch $BATCH --accum $ACCUM --beta 1.0 --base $BASE \
    --val-ring $VRING --feed-timeout 1800 \
    --out $OUT --ckpt-every 2000 $R >> $D/train.log 2>&1
  echo "$(date +%T) SUP: trainer exited (restart $N, resume='$ck')" >> $D/sup.log
  N=$((N+1)); sleep 20
done

if [ -f ${OUT}_final.pt ]; then
  echo "$(date +%T) SUP: trainer DONE -> auto-eval" >> $D/sup.log
  M="final=${OUT}_final.pt:$BASE"
  [ -f ${OUT}_best.pt ] && M="$M best=${OUT}_best.pt:$BASE"
  CUDA_VISIBLE_DEVICES=$GPU python3 $(dirname "$0")/eval_students.py \
    --crops $CROPS --out ${OUT}_eval.json $M >> $D/eval.log 2>&1
  echo "$(date +%T) SUP: eval -> ${OUT}_eval.json" >> $D/sup.log
  ck=${OUT}_best.pt; [ -f $ck ] || ck=${OUT}_final.pt
  CUDA_VISIBLE_DEVICES=$GPU python3 $(dirname "$0")/trace_eval_run.py \
    --ckpt $ck:$BASE --crops $CROPS --out ${OUT}_trace_eval.json >> $D/eval.log 2>&1
  echo "$(date +%T) SUP: trace-eval -> ${OUT}_trace_eval.json" >> $D/sup.log
fi
