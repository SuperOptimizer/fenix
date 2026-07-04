#!/bin/sh
# tools/inkhunt/refine_transform.sh — refine a cross-scan transform by coordinate descent
# on the TRUE objective: surf-qc delta of a transformed reference segment against the new
# volume. This is what took PHercParis3 from the phase-correlation near-miss (delta -3.7)
# to native-trace quality (+14.9): the correlation gets you in-frame, the descent walks
# the last tens of voxels on real evidence.
#   refine_transform.sh <fenix> <ref.obj> <new-cache@zarr> <scale> <tz> <ty> <tx> <out.json>
# scale/t are the LOD0->LOD0 starting affine (ZYX t, in new-volume voxels).
set -u
F=$1; OBJ=$2; CACHE=$3; S=$4; TZ=$5; TY=$6; TX=$7; OUT=$8
eval_at() {
  $F import-obj "$OBJ" /tmp/refine_$$.fxsurf grid=8 "affine=$S,0,0,$1,0,$S,0,$2,0,0,$S,$3" >/dev/null 2>&1 || { echo "-99"; return; }
  d=$($F surf-qc "$CACHE" /tmp/refine_$$.fxsurf k=40 off=12 min_delta=5 2>/dev/null | grep -o "delta [+-][0-9.]*" | awk '{print $2}')
  [ -n "$d" ] && echo "$d" || echo "-99"
}
BEST=$(eval_at $TZ $TY $TX)
echo "refine: start t=($TZ,$TY,$TX) delta $BEST"
for STEP in 128 64 32 16 8 4 2; do
  IMPROVED=1
  while [ "$IMPROVED" = "1" ]; do
    IMPROVED=0
    for AX in z y x; do
      for SGN in 1 -1; do
        NZ=$TZ; NY=$TY; NX=$TX
        D=$((STEP * SGN))
        case $AX in z) NZ=$((TZ + D));; y) NY=$((TY + D));; x) NX=$((TX + D));; esac
        V=$(eval_at $NZ $NY $NX)
        if [ "$(awk "BEGIN{print ($V > $BEST + 0.05) ? 1 : 0}")" = "1" ]; then
          TZ=$NZ; TY=$NY; TX=$NX; BEST=$V; IMPROVED=1
          echo "refine: step $STEP -> t=($TZ,$TY,$TX) delta $BEST"
        fi
      done
    done
  done
done
rm -f /tmp/refine_$$.fxsurf
echo "refine: FINAL t=($TZ,$TY,$TX) delta $BEST"
printf '{\n  "transformation_matrix": [\n    [%s, 0.0, 0.0, %s],\n    [0.0, %s, 0.0, %s],\n    [0.0, 0.0, %s, %s],\n    [0.0, 0.0, 0.0, 1.0]\n  ],\n  "note": "lod0->lod0 XYZ rows; refined by surf-qc coordinate descent (delta %s)"\n}\n' \
  "$S" "$TX" "$S" "$TY" "$S" "$TZ" "$BEST" > "$OUT"
echo "refine: wrote $OUT"
# exit 0 iff the frame is proven
awk "BEGIN{exit !($BEST >= 5)}"
