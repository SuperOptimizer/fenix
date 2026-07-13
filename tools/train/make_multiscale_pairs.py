#!/usr/bin/env python3
"""Transform a 2.4-canonical pairs file into a coarse-canon (4.8/9.6) variant.

Native-2.4 sources are pointed at the zarr pyramid level whose pitch equals the
target canon (level 1 for 4.8, level 2 for 9.6): um=canon, msc=1/2^k (mesh coords
stay on their trace grid), scale exactly 1 -> no CT resampling and 8x/64x less WAN.
Odd-grid scrolls (um= already on the line, e.g. 0500P2 4.317, 0172 7.91) pick the
pyramid level closest to canon from THEIR source grid and carry the residual
resample (e.g. 0172 7.91 -> canon=9.6 is scale 0.82 at level 0).
Caches get level-suffixed paths (one writer per cache file).

Usage: make_multiscale_pairs.py <pairs.txt> <canon: 4.8|9.6> <out.txt>
"""
import sys, re

pairs, canon, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
lines = []
for line in open(pairs):
    toks = line.split()
    if not toks:
        continue
    um = 2.4
    for t in toks:
        if t.startswith("um="):
            um = float(t[3:])
    toks = [t for t in toks if not t.startswith(("um=", "canon=", "msc="))]
    # deepest pyramid level from THIS source grid that stays at or above canon pitch
    k = 0
    while um * 2 ** (k + 1) <= canon * 1.001:
        k += 1
    if k > 0:
        toks[1] = re.sub(r"\.fxvol@", f"_L{k}.fxvol@", toks[1])
        toks[1] = re.sub(r"/0$", f"/{k}", toks[1])
    lvl_um = um * 2 ** k
    toks.append(f"um={canon if abs(lvl_um - canon) / canon < 0.002 else lvl_um:g}")
    if k > 0:
        toks.append(f"msc={1 / 2 ** k:g}")
    toks.append(f"canon={canon:g}")
    lines.append(" ".join(toks))
open(out, "w").write("\n".join(lines) + "\n")
print(f"{len(lines)} pairs -> {out} (canon={canon})")
