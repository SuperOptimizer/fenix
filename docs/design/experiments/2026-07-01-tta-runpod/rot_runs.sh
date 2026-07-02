#!/bin/bash
set -u
W=/dev/shm/fenix/models/surface_recto_3dunet/surface.fxweights
cd /dev/shm
run() { # run <crop> <name> <tta> <rots>
  local crop=$1 name=$2 tta=$3 rots=$4
  local out=/dev/shm/tta/${crop%.fxvol}_${name}.fxvol
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  local t0=$(date +%s.%N)
  /root/fenix predict-surface "$crop" "$W" "$out" 256 0.5 "$tta" 3 "rots=$rots" > /dev/null 2>&1
  local rc=$?
  echo "TIMING crop=$crop combo=$name walltime=$(echo "$(date +%s.%N) $t0" | awk '{printf "%.1f", $1-$2}')s rc=$rc"
}
for crop in crop512.fxvol crop512b.fxvol; do
  run "$crop" r4   0 "0,22.5,45,67.5"
  run "$crop" r8   0 "0,11.25,22.5,33.75,45,56.25,67.5,78.75"
  run "$crop" t8r4 8 "0,22.5,45,67.5"
done
echo ROT_RUNS_DONE
