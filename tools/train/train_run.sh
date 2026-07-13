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
# Feed scaling (measured 2026-07-11, warm cache, 32c box, live-training contention):
# ~linear in THREADS, superlinear in ECHO — t8/e1 2.0 p/s, t16/e2 19.8, t16/e4 45.9,
# t24/e2 35.4, t24/e4 77.5. A 5060Ti trainer eats ~6 p/s (t12/e1 suffices); a 5090
# wants ~25 -> t16/e2 or t24/e2. Echo's cost is correlated samples (same draw,
# independent augs) — raise THREADS first, then ECHO.
THREADS=12; ECHO=1
# Sharded-source mode (dct3d exports): SHARD_GRID=1024 LOCALITY=64 makes each feeder
# cluster drain one downloaded shard from disk (measured 7.8 draws/s COLD at t8 vs
# 2-4 on raw chunks). 0 = off (raw chunk sources).
SHARD_GRID=0; LOCALITY=32
# Feed-cache knobs (measured 2026-07-12 on a shared box, so magnitudes are noisy but
# directions were consistent across 9 drains: 19->26 draws/s from CACHE_MB 4096->16384,
# ->32 with PREFETCH 512, locality 32 added more): bigger decoded-block cache = fewer
# DCT re-decodes; deeper prefetch = better cold-region overlap; locality 32 doubles
# fetch reuse per cluster (distribution note: doubles clustered run length, members
# still ±spread around surface-weighted centers). Size CACHE_MB to the box: it is
# per-VOLUME (5 caches x 16GB won't fit a 64GB box under training — the budget is a
# cap, actual use tracks the hot set).
FEED_CACHE_MB=16384; PREFETCH=512
BETA=1.0; AUG=2   # loss/aug recipe knobs. M12 note: the M10 recipe (BETA=1.0 AUG=2,
                  # 60k, base-32) was CONVICTED by the studentF replication (recall2
                  # 0.204/0.236 vs studentG's 0.252 at 6x less compute) — the proven
                  # M8 recipe is BETA=0.3 AUG=1 STEPS=10000 BASE=16.
FP8=0   # BROKEN — forensics only (see train.py --fp8 help)
NGPU=1  # >1: DDP via torchrun — one feeder+ring PER RANK (<ring>.rN), grads allreduce,
        # rank 0 owns val/EMA/checkpoints. Effective batch = NGPU*BATCH*ACCUM.
# Measured matrix 2026-07-12 (5060 Ti, batch4 dice+CE, sep-verified): bf16+CL+compile
# 592ms vs 679 plain bf16 (1.15x); CL alone HURTS eager (944ms) — only enable together.
COMPILE=1; CHANNELS_LAST=1
for a in "$@"; do eval "$a"; done
D=$(dirname $OUT); mkdir -p $D
# TOOLS: where train.py/eval_students.py live. dirname $0 breaks the moment someone copies
# this script elsewhere to tweak knobs (measured: a sed'd copy in /workspace looked for
# /workspace/train.py and crash-looped 24 times) — override with TOOLS_DIR in that case.
TOOLS=${TOOLS_DIR:-$(dirname "$0")}
FENIX=${FENIX_BIN:-$TOOLS/../../build-release/fenix}
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
if [ "$NGPU" -gt 1 ]; then
  r=0
  while [ $r -lt $NGPU ]; do
    feed_loop "$PAIRS" "$RING.r$r" $D/feedM.r$r.log \
      patch=128 slots=32 threads=$THREADS echo=$ECHO seed=$((42+r)) aug=$AUG disk_mb=131072 locality=$LOCALITY shard_grid=$SHARD_GRID cache_mb=$FEED_CACHE_MB prefetch=$PREFETCH &
    r=$((r+1))
  done
else
  feed_loop "$PAIRS" "$RING" $D/feedM.log \
    patch=128 slots=32 threads=$THREADS echo=$ECHO seed=42 aug=$AUG disk_mb=131072 locality=$LOCALITY shard_grid=$SHARD_GRID cache_mb=$FEED_CACHE_MB prefetch=$PREFETCH &
fi
feed_loop "$VPAIRS" "$VRING" $D/feedV.log \
  patch=128 slots=8 threads=4 seed=777 aug=0 disk_mb=131072 &

N=0
while [ ! -f ${OUT}_final.pt ] && [ $N -lt 80 ]; do
  if pgrep -f "train.py --ring $RING" >/dev/null 2>&1; then sleep 60; continue; fi
  ck=$(ls -t ${OUT}_step*.pt 2>/dev/null | head -1)
  [ -n "$ck" ] && R="--resume $ck" || R=""
  if [ "$NGPU" -gt 1 ]; then LAUNCH="torchrun --standalone --nproc-per-node $NGPU $TOOLS/train.py"; \
  else LAUNCH="python3 $TOOLS/train.py"; fi
  CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $LAUNCH \
    --ring $RING --steps $STEPS --batch $BATCH --accum $ACCUM --beta $BETA --base $BASE \
    $([ "$FP8" = "1" ] && echo --fp8) \
    $([ "$COMPILE" = "1" ] && echo --compile) \
    $([ "$CHANNELS_LAST" = "1" ] && echo --channels-last) \
    --val-ring $VRING --feed-timeout 1800 \
    --out $OUT --ckpt-every 2000 $R >> $D/train.log 2>&1
  echo "$(date +%T) SUP: trainer exited (restart $N, resume='$ck')" >> $D/sup.log
  N=$((N+1)); sleep 20
done

if [ -f ${OUT}_final.pt ]; then
  echo "$(date +%T) SUP: trainer DONE -> auto-eval" >> $D/sup.log
  M="final=${OUT}_final.pt:$BASE"
  [ -f ${OUT}_best.pt ] && M="$M best=${OUT}_best.pt:$BASE"
  CUDA_VISIBLE_DEVICES=$GPU python3 $TOOLS/eval_students.py \
    --crops $CROPS --out ${OUT}_eval.json $M >> $D/eval.log 2>&1
  echo "$(date +%T) SUP: eval -> ${OUT}_eval.json" >> $D/sup.log
  ck=${OUT}_best.pt; [ -f $ck ] || ck=${OUT}_final.pt
  CUDA_VISIBLE_DEVICES=$GPU python3 $TOOLS/trace_eval_run.py \
    --ckpt $ck:$BASE --crops $CROPS --out ${OUT}_trace_eval.json >> $D/eval.log 2>&1
  echo "$(date +%T) SUP: trace-eval -> ${OUT}_trace_eval.json" >> $D/sup.log
fi
