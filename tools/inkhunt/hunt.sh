#!/bin/sh
# tools/inkhunt/hunt.sh — the automated segment-and-ink hunt (one scroll per invocation).
# For every mirrored segment: import (with the cross-scan transform) -> surf-qc frame
# gate -> render-layers from the NEW 2.4um volume -> predict-ink -> max-projection
# review JPEG. Segments failing the frame gate are listed, not silently skipped.
#   hunt.sh <segdir> <transform.json> <pre_scale> <post_scale> <new-zarr-url> <workdir> <ink.fxweights>
# example (Paris3):
#   hunt.sh /workspace/ink/PHercParis3 /workspace/reg/paris3_old_to_new.json 0.0625 32 \
#     https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis3/volumes/20260427095331-2.400um-0.2m-78keV-masked.zarr/0 \
#     /workspace/hunt/paris3 /workspace/models/ink.fxweights
set -u
F=/workspace/fenix/build-release/fenix
SEGDIR=$1; XFORM=$2; PRE=$3; POST=$4; ZARR=$5; WORK=$6; WEIGHTS=$7
mkdir -p "$WORK"
CACHE="$WORK/ct_cache.fxvol@$ZARR"
for d in "$SEGDIR"/*/; do
  s=$(basename "$d")
  obj="$d/$s.obj"
  [ -f "$obj" ] || continue
  out="$WORK/$s"
  [ -f "$out.ink.jpg" ] && continue          # resumable
  echo "=== $s $(date -u +%H:%M:%S)"
  $F import-obj "$obj" "$out.fxsurf" grid=8 transform="$XFORM" pre_scale="$PRE" post_scale="$POST" || { echo "IMPORT_FAIL $s"; continue; }
  if ! $F surf-qc "$CACHE" "$out.fxsurf" k=80 off=12 min_delta=3; then
    echo "FRAME_FAIL $s (delta below gate — listed for review)"; continue
  fi
  # 65 layers at NATIVE spacing: both converted ink models are canonical-2um-era
  # (ink_3d_dino_guided 06-30, ink_canonical_2um) — they want ~2.4um voxels, step=1.
  # (step=3.296 is only for 7.91um-era models like the 2023 timesformers.)
  $F render-layers "$CACHE" "$out.fxsurf" "$out.stack.fxvol" layers=65 step=1 q=8 || { echo "RENDER_FAIL $s"; continue; }
  $F predict-ink "$out.stack.fxvol" "$WEIGHTS" "$out.inkprob.fxvol" 128 0.5 || { echo "INK_FAIL $s"; continue; }
  $F project "$out.inkprob.fxvol" "$out.ink.jpg" mode=max
  $F project "$out.stack.fxvol" "$out.tex.jpg" mode=mean   # papyrus texture alongside
  echo "OK $s"
done
echo HUNT_DONE
