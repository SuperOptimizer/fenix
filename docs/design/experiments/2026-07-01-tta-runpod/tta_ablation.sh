#!/bin/bash
# TTA ablation for the distillation teacher: which augmentation combo, at what cost.
# Combos run per crop; t48ms (tta=48 + scales=1,1.2) is the maximal-ensemble reference.
set -u
W=/dev/shm/fenix/models/surface_recto_3dunet/surface.fxweights
mkdir -p /dev/shm/tta
cd /dev/shm
run() { # run <crop> <name> <tta> [scales]
  local crop=$1 name=$2 tta=$3 scales=${4:-}
  local out=/dev/shm/tta/${crop%.fxvol}_${name}.fxvol
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  local t0=$(date +%s.%N)
  if [ -n "$scales" ]; then
    /root/fenix predict-surface "$crop" "$W" "$out" 256 0.5 "$tta" 3 "scales=$scales" > /dev/null 2>&1
  else
    /root/fenix predict-surface "$crop" "$W" "$out" 256 0.5 "$tta" 3 > /dev/null 2>&1
  fi
  local rc=$?
  local t1=$(date +%s.%N)
  echo "TIMING crop=$crop combo=$name walltime=$(echo "$t1 $t0" | awk '{printf "%.1f", $1-$2}')s rc=$rc"
}
for crop in crop512.fxvol crop512b.fxvol; do
  run "$crop" t0    0
  run "$crop" t8    8
  run "$crop" t24   24
  run "$crop" t48   48
  run "$crop" s12   0  1.2
  run "$crop" ms    0  1,1.2
  run "$crop" t8ms  8  1,1.2
  run "$crop" t48ms 48 1,1.2
done
echo ALL_RUNS_DONE
