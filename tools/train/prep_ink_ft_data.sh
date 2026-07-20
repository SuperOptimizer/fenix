#!/bin/bash
# prep_ink_ft_data.sh — build the (raw, q32, teacher) region triples for
# finetune_ink_compression.py. Per region: ingest raw CT (q=2, ~lossless) from the
# open-data zarr, transcode to q=32, teacher-predict ink on the raw CT (net=ink tta=8),
# export all three as .npy. Regions with ~zero teacher ink mass are reported so they can
# be dropped. Idempotent per region (skips existing .npy triples).
#
#   FENIX=~/fenix/build-release/fenix DATA=~/inkft \
#     bash prep_ink_ft_data.sh "z y x name" "z y x name" ...
# Defaults: 6 × 768³ PHerc0332 regions around the known-ink band of the 07-19 smokes.
set -eu
FENIX=${FENIX:-$HOME/fenix/build-release/fenix}
DATA=${DATA:-$HOME/inkft}
SRC=${SRC:-https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0332/volumes/20251211183505-2.399um-0.2m-78keV-masked.zarr}
D=${D:-768}
TTA=${TTA:-8}
mkdir -p "$DATA"

REGIONS=("$@")
if [ ${#REGIONS[@]} -eq 0 ]; then
  REGIONS=(
    "16384 7168 7168 r16384a"
    "16384 6400 8704 r16384b"
    "14336 7680 6656 r14336a"
    "18432 6656 7680 r18432a"
    "20480 7936 7168 r20480a"
    "12288 7168 8192 r12288a"
  )
fi

for spec in "${REGIONS[@]}"; do
  read -r Z Y X NAME <<<"$spec"
  if [ -e "$DATA/$NAME.teacher.npy" ]; then echo "== $NAME: exists, skip"; continue; fi
  echo "== $NAME: z$Z y$Y x$X +${D}^3"
  time "$FENIX" ingest-zarr "$SRC" 0 "$Z" "$Y" "$X" "$D" "$D" "$D" "$DATA/$NAME.raw.fxvol" q=2
  time "$FENIX" transcode "$DATA/$NAME.raw.fxvol" "$DATA/$NAME.q32.fxvol" 32
  time "$FENIX" predict-scroll "$DATA/$NAME.raw.fxvol" 0 "$HOME/ink3d.fxweights" \
      "$DATA/$NAME.teacher.fxvol" net=ink tta=$TTA batch=4 patch=256 mode=global q=32
  "$FENIX" export-npy "$DATA/$NAME.raw.fxvol" "$DATA/$NAME.raw.npy"
  "$FENIX" export-npy "$DATA/$NAME.q32.fxvol" "$DATA/$NAME.q32.npy"
  "$FENIX" export-npy "$DATA/$NAME.teacher.fxvol" "$DATA/$NAME.teacher.npy"
  python3 - "$DATA/$NAME.teacher.npy" <<'EOF'
import numpy as np, sys
t = np.load(sys.argv[1], mmap_mode="r")
frac = float((t[::4, ::4, ::4] > 25).mean())
print(f"ink-positive fraction (>0.1 prob, /64 sampled): {frac:.5f}" + ("  << NEAR-ZERO, consider dropping" if frac < 1e-4 else ""))
EOF
  rm -f "$DATA/$NAME.teacher.fxvol.ctcache.fxvol"  # predict-scroll side product, not needed
done
echo PREP_ALL_DONE
