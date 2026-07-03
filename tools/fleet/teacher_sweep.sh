#!/bin/sh
# Teacher-sweep fleet worker: claim volume-block items by static shard, band-ingest the CT,
# run band-filtered resumable tta predict.
#   teacher_sweep.sh <sweep.txt> <shard> <nshards> [tta=8] [band_r=384] [outdir=/workspace/teacher]
# Needs the fxsurf corpus at $FXROOT (default /workspace/corpus — pods get a tarball; the
# corpus is gitignored). Sharding is deterministic (line-index % nshards) — no coordination.
# Re-running resumes: .done markers per item; a killed mid-item predict resumes from its
# checkpoint (crash-safe per-tile). Engine: /root/surface_256b1.plan (TTA runs members
# singly — a b3 engine would pad 1->3). pod_bootstrap.sh builds it.
set -eu
SWEEP=$1
SHARD=$2
NSHARDS=$3
TTA=${4:-8}
BAND_R=${5:-384}
OUT=${6:-/workspace/teacher}
FXROOT=${FXROOT:-/workspace/corpus}
F=/workspace/fenix/build-release/fenix
ENGINE=/root/surface_256b1.plan
mkdir -p "$OUT"

i=0
while IFS=" " read -r name url um z0 y0 x0 D H W rels; do
  i=$((i + 1))
  if [ $(((i - 1) % NSHARDS)) -ne "$SHARD" ]; then continue; fi
  [ -e "$OUT/$name.done" ] && continue
  echo "[$i] $name ${D}x${H}x${W} @ $z0,$y0,$x0"
  SURFS=$(echo "$rels" | tr ',' '\n' | sed "s|^|$FXROOT/|" | paste -sd, -)
  CT="$OUT/$name.ct.fxvol"
  if [ ! -e "$CT" ]; then
    "$F" ingest-band "$url" 0 "$CT.part" "$SURFS" "$z0" "$y0" "$x0" "$D" "$H" "$W" "band_r=$BAND_R" q=8
    mv "$CT.part" "$CT"
  fi
  "$F" predict-surface "$CT" "$ENGINE" "$OUT/$name.teacher.fxvol" 256 0.5 "$TTA" \
    "band=$SURFS" "band_off=$z0,$y0,$x0" "band_r=$BAND_R"
  rm -f "$CT"
  : > "$OUT/$name.done"
done < "$SWEEP"
echo "shard $SHARD/$NSHARDS complete"
