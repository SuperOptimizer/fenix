#!/bin/sh
# tools/inkhunt/hunt.sh — the automated segment-and-ink hunt (one scroll per invocation).
# For every mirrored segment: import (with the cross-scan transform) -> surf-qc frame
# gate -> render-layers from the NEW 2.4um volume -> ALL THREE released ink models ->
# consensus + review JPEGs. Segments failing the frame gate are listed, not skipped
# silently. Per-output resume guards: re-running backfills missing outputs only.
#   hunt.sh <segdir> <transform.json> <pre_scale> <post_scale> <new-zarr-url> <workdir> <ink.fxweights>
# Companion weights are found next to <ink.fxweights>: ink2d.fxweights (r152), ink50.fxweights (r50).
# example (Paris3):
#   hunt.sh /workspace/ink/PHercParis3 /workspace/reg/paris3_old_to_new.json 0.0625 32 \
#     https://vesuvius-challenge-open-data.s3.amazonaws.com/PHercParis3/volumes/20260427095331-2.400um-0.2m-78keV-masked.zarr/0 \
#     /workspace/hunt/paris3 /workspace/models/ink.fxweights
set -u
F=/workspace/fenix/build-release/fenix
SEGDIR=$1; XFORM=$2; PRE=$3; POST=$4; ZARR=$5; WORK=$6; WEIGHTS=$7
WDIR=$(dirname "$WEIGHTS")
W2D="$WDIR/ink2d.fxweights"
W50="$WDIR/ink50.fxweights"
mkdir -p "$WORK"
CACHE="$WORK/ct_cache.fxvol@$ZARR"
for d in "$SEGDIR"/*/; do
  s=$(basename "$d")
  obj="$d/$s.obj"
  [ -f "$obj" ] || continue
  out="$WORK/$s"
  echo "=== $s $(date -u +%H:%M:%S)"
  # per-segment refined transform (from refine_transform.sh) beats the global one;
  # its presence also clears a stale FRAME_FAIL so the segment is re-gated
  SXF="$XFORM"
  if [ -f "$WORK/$s.transform.json" ]; then
    SXF="$WORK/$s.transform.json"
    [ -f "$out.FRAME_FAIL" ] && rm -f "$out.FRAME_FAIL" "$out.fxsurf"
  fi
  [ -f "$out.FRAME_FAIL" ] && continue
  if [ ! -f "$out.fxsurf" ]; then
    $F import-obj "$obj" "$out.fxsurf" grid=8 transform="$SXF" pre_scale="$PRE" post_scale="$POST" || { echo "IMPORT_FAIL $s"; continue; }
    if ! $F surf-qc "$CACHE" "$out.fxsurf" k=80 off=12 min_delta=3; then
      echo "FRAME_FAIL $s (delta below gate — listed for review)"; touch "$out.FRAME_FAIL"; continue
    fi
  fi
  # 65 layers at NATIVE spacing: dino + r152 are canonical-2um-era -> step=1.
  # (step=3.296 is only for 7.91um-era models like the 2023 timesformers.)
  if [ ! -f "$out.stack.fxvol" ]; then
    $F render-layers "$CACHE" "$out.fxsurf" "$out.stack.fxvol" layers=65 step=1 q=8 || { echo "RENDER_FAIL $s"; continue; }
  fi
  [ -f "$out.tex.jpg" ] || $F project "$out.stack.fxvol" "$out.tex.jpg" mode=mean
  # (1) dino 3D UNet -> per-voxel prob -> max projection
  if [ ! -f "$out.ink.jpg" ]; then
    $F predict-ink "$out.stack.fxvol" "$WEIGHTS" "$out.inkprob.fxvol" 128 0.5 || { echo "INK_FAIL $s"; continue; }
    $F project "$out.inkprob.fxvol" "$out.ink.jpg" mode=max
    rm -f "$out.inkprob.fxvol"
  fi
  # (2) r152 (ink_canonical_2um) on the same 65-layer stack
  if [ -f "$W2D" ] && [ ! -f "$out.ink2d.jpg" ]; then
    $F predict-ink2d "$out.stack.fxvol" "$W2D" "$out.ink2d.jpg" || echo "INK2D_FAIL $s"
  fi
  # (3) r50 (resnet50_3um): 3um-native -> its OWN 18-layer stack at step=1.25
  if [ -f "$W50" ] && [ ! -f "$out.ink50.jpg" ]; then
    if [ ! -f "$out.stack50.fxvol" ]; then
      $F render-layers "$CACHE" "$out.fxsurf" "$out.stack50.fxvol" layers=18 step=1.25 q=8 || echo "RENDER50_FAIL $s"
    fi
    [ -f "$out.stack50.fxvol" ] && { $F predict-ink2d "$out.stack50.fxvol" "$W50" "$out.ink50.jpg" net=r50 || echo "INK50_FAIL $s"; }
    rm -f "$out.stack50.fxvol"
  fi
  # (4) cross-model consensus (geometric mean of whatever maps exist)
  python3 /workspace/fenix/tools/inkhunt/consensus.py "$out" || echo "CONS_FAIL $s"
  echo "OK $s"
done
echo HUNT_DONE
