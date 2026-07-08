#!/bin/sh
# vast_onstart.sh — first-boot entrypoint of the fenix-vast image (Dockerfile.vast).
# Everything general is baked; this does only the machine-locked + job-specific parts:
#   1. build the TensorRT engine for THIS GPU arch from the baked ONNX (idempotent, ~1-2 min)
#   2. if SWEEP_URL is set: fetch the manifest and run the shard worker (resumable — an
#      interrupted instance that resumes just re-enters here and skips .done items)
#   3. else: idle (dev box — ssh in and drive by hand)
#
# Env (vast `--env '-e K=V ...'`): SWEEP_URL, SHARD, NSHARDS, TTA (default 8), BAND_R,
# ENGINE_BATCH (default 1 — TTA runs members singly; b3 would pad 1->3).
set -eu
cd /opt/fenix
OUT=${OUT:-/workspace/teacher}
mkdir -p /workspace "$OUT"

EB=${ENGINE_BATCH:-1}
ENGINE=/workspace/surface_256b${EB}.plan
if [ ! -e "$ENGINE" ]; then
  echo "[onstart] building TRT engine for $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  python3 tools/ml-export/build_engine.py /opt/fenix/surface.onnx "$ENGINE.part" 256 "$EB"
  mv "$ENGINE.part" "$ENGINE"
fi

if [ -n "${SWEEP_URL:-}" ]; then
  curl -sf "$SWEEP_URL" -o /workspace/sweep.txt
  echo "[onstart] shard ${SHARD:?}/${NSHARDS:?} on $(wc -l < /workspace/sweep.txt) items"
  ENGINE=$ENGINE exec sh tools/fleet/teacher_sweep.sh /workspace/sweep.txt \
    "$SHARD" "$NSHARDS" "${TTA:-8}" "${BAND_R:-384}" "$OUT"
fi
echo "[onstart] no SWEEP_URL — idling (dev mode)"
exec sleep infinity
