#!/usr/bin/env bash
# One-scroll GT grade -> (repair) -> accept-set export pipeline. Consolidates the four
# /tmp scroll_pipeline*.sh variants that ran the 2026-07 multi-scroll campaign.
#
# Usage: grade_scroll.sh <scroll> <zarr-root-url (no /0)> [options]
#   --um <f>       voxel um of this scroll's grid when != 2.4 (emitted on pair lines)
#   --local        meshes exist only as $GTQC/fxsurf/<scroll>__*.fxsurf (OBJ imports,
#                  repaired surfaces): crop/orientation read a fenix export-tifxyz dump
#                  instead of the bucket tifxyz from corpus_import.txt
#   --no-align     alignment axis unmeasurable in this domain (FAILED fault injection —
#                  run tools/labelqc/fault_inject.py on every new scan resolution/optics
#                  FIRST): skip crop sweep + repair, scorecard caps at B
#   --no-dedup     hand-labeled distinct wrap identities: DISAGREE pairs are adjacent
#                  sheets, never dropped
#
# Env: GTQC_DIR (default /tmp/gtqc), FENIX_BIN (default build-release/fenix)
set -u
S=$1; Z=$2; shift 2
UM=""; LOCAL=0; NOALIGN=0; NODEDUP=""
while [ $# -gt 0 ]; do case $1 in
  --um) UM=$2; shift 2;;
  --local) LOCAL=1; shift;;
  --no-align) NOALIGN=1; shift;;
  --no-dedup) NODEDUP="--no-dedup"; shift;;
  *) echo "unknown arg $1" >&2; exit 1;;
esac; done

G=${GTQC_DIR:-/tmp/gtqc}
F=${FENIX_BIN:-$(dirname "$0")/../../../build-release/fenix}
LQ=$(dirname "$0")/..
CACHE=/home/forrest/gtqc-caches/ct_${S}.fxvol
LOG=$G/${S}_pipeline.log
TIFD=$G/tifxyz_local
mkdir -p $G/cropcards $G/scorecards $TIFD
echo "$(date +%T) [$S] pipeline start (local=$LOCAL no-align=$NOALIGN um=${UM:-2.4})" >> $LOG

# resolve each segment's tifxyz dir (bucket URL or local export) + name
seg_tifs() {
  if [ $LOCAL = 1 ]; then
    for f in $G/fxsurf/${S}__*.fxsurf; do
      seg=$(basename $f .fxsurf)
      [ -f $TIFD/$seg/x.tif ] || $F export-tifxyz $f $TIFD/$seg >> $LOG 2>&1
      echo "$seg $TIFD/$seg"
    done
  else
    grep "^$S	" $G/corpus_import.txt | cut -f3 | while read P; do
      seg="${S}__$(echo "$P" | sed -E "s#$S/segments/([^/]+)/.*#\1#")"
      echo "$seg https://vesuvius-challenge-open-data.s3.amazonaws.com/${P%/}"
    done
  fi
}

# 1) crop-QC sweep (alignment axis) — skipped in no-align domains
if [ $NOALIGN = 0 ]; then
  seg_tifs | while read seg TIF; do
    out="$G/cropcards/${seg}.json"
    [ -f "$out" ] && continue
    timeout 1200 python3 $LQ/crop_qc.py --tifxyz "$TIF" --zarr "$Z" --crops 8 --uv 48 --out "$out" >> $LOG 2>&1
    echo "$(date +%T) [$S] crop-qc $seg $( [ -f "$out" ] && echo ok || echo FAIL)" >> $LOG
  done
  echo "$(date +%T) [$S] crop sweep done: $(ls $G/cropcards/${S}__*.json 2>/dev/null | wc -l)" >> $LOG
fi

# 2) mesh health (CT-free) for any segment not yet in the store
for f in $G/fxsurf/${S}__*.fxsurf; do
  seg=$(basename $f .fxsurf)
  grep -q "$seg.fxsurf" $G/meshqual.jsonl 2>/dev/null || $F mesh-qual $f --json >> $G/meshqual.jsonl 2>/dev/null
done

# 3) umbilicus axis from this scroll's meshes
ARGS=""
for f in $G/fxsurf/${S}__*.fxsurf; do ARGS="$ARGS surf=$f"; done
$F umbilicus $ARGS out=$G/${S}_axis.toml band=1024 stride=4 >> $LOG 2>&1
echo "$(date +%T) [$S] umbilicus $( [ -f $G/${S}_axis.toml ] && echo ok || echo FAIL)" >> $LOG

# 4) orientation (writes its own well-formed jsonl — never shell-assembled JSON)
seg_tifs | while read seg TIF; do
  grep -q "\"$seg\"" $G/orientation.jsonl 2>/dev/null && continue
  timeout 300 python3 $LQ/orientation_check.py --tifxyz "$TIF" --axis $G/${S}_axis.toml \
    --jsonl $G/orientation.jsonl --seg "$seg" >> $LOG 2>&1
done
echo "$(date +%T) [$S] orientation done" >> $LOG

# 5) consist (giants config: pts_mb=16 undersamples big-mesh pairs at near=6)
timeout 7200 $F surf-consist $G/fxsurf/${S}__*.fxsurf k=1500 near=6 pts_mb=128 > $G/consist_${S}.txt 2>&1
echo "$(date +%T) [$S] consist: $(grep -c 'pair' $G/consist_${S}.txt) pairs" >> $LOG

# 6) scorecards (+ borderline escalation at 16 crops in aligned domains)
NA=""; [ $NOALIGN = 1 ] && NA="--no-align"
python3 $LQ/scorecard.py --scroll $S --consist $G/consist_${S}.txt ${UM:+--um $UM} $NA >> $LOG 2>&1
if [ $NOALIGN = 0 ]; then
  python3 -c "
import json,glob
for p in glob.glob('$G/scorecards/${S}__*.card.json'):
    c=json.load(open(p))
    if c['grade'].endswith('?'): print(c['segment'])
" | while read seg; do
    TIF=$(seg_tifs | grep "^$seg " | cut -d' ' -f2)
    [ -z "$TIF" ] && continue
    echo "$(date +%T) [$S] escalate $seg (16 crops)" >> $LOG
    timeout 2400 python3 $LQ/crop_qc.py --tifxyz "$TIF" --zarr "$Z" --crops 16 --uv 48 \
      --out "$G/cropcards/${seg}.json" >> $LOG 2>&1
  done
  python3 $LQ/scorecard.py --scroll $S --consist $G/consist_${S}.txt ${UM:+--um $UM} >> $LOG 2>&1

  # 7) repair loop (aligned domains only — no-align snap range spans the inter-sheet gap)
  python3 $LQ/repair_corpus.py --scroll $S --ct "$Z/0" --cache $CACHE >> $LOG 2>&1
fi

# 8) accept-set export
python3 $LQ/export_acceptset.py --scroll $S --ct "$CACHE@$Z/0" ${UM:+--um $UM} $NODEDUP \
  --out $G/pairs_${S}.txt >> $LOG 2>&1
echo "$(date +%T) [$S] PIPELINE DONE" >> $LOG
