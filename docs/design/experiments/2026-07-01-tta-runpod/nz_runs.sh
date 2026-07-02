#!/bin/bash
set -u
W=/dev/shm/fenix/models/surface_recto_3dunet/surface.fxweights
cd /dev/shm
run() { # crop name extra_args
  local crop=$1 name=$2; shift 2
  local out=/dev/shm/tta/${crop%.fxvol}_${name}.fxvol
  [ -f "$out" ] && { echo "SKIP $out"; return; }
  local t0=$(date +%s.%N)
  /root/fenix predict-surface "$crop" "$W" "$out" 256 0.5 0 3 "$@" > /dev/null 2>&1
  echo "TIMING crop=$crop combo=$name walltime=$(echo "$(date +%s.%N) $t0" | awk '{printf "%.1f", $1-$2}')s rc=$?"
}
for crop in crop512.fxvol crop512b.fxvol; do
  run "$crop" n4    noise=4 nsigma=0.1
  run "$crop" n8    noise=8 nsigma=0.1
  run "$crop" o3    offsets=3
  { o=/dev/shm/tta/${crop%.fxvol}_t8n4.fxvol; [ -f "$o" ] || /root/fenix predict-surface "$crop" "$W" "$o" 256 0.5 8 3 noise=4 nsigma=0.1 >/dev/null 2>&1; echo "TIMING crop=$crop combo=t8n4 rc=$?"; }
done
echo NZ_RUNS_DONE
