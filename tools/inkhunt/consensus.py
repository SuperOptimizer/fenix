#!/usr/bin/env python3
"""consensus.py — cross-model ink consensus for one hunt segment.

Usage: consensus.py <workdir/segment-prefix>

Loads whichever of <prefix>.ink.jpg (dino 3D), <prefix>.ink2d.jpg (r152),
<prefix>.ink50.jpg (r50) exist, per-map percentile-normalizes to [0,1] (the three
models are calibrated differently — raw levels are not comparable), resizes to the
largest map, and writes <prefix>.cons.jpg = the GEOMETRIC mean. Geometric (not
arithmetic) so a region needs support from EVERY model — agreement between
independent architectures is the strongest cheap ink signal; one model's lone
hallucination multiplies toward zero.
"""
import sys

import numpy as np
from PIL import Image


def load_norm(path):
    try:
        a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    except FileNotFoundError:
        return None
    lo, hi = np.percentile(a, 1.0), np.percentile(a, 99.5)
    if hi - lo < 1e-3:
        return np.zeros_like(a)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0)


def main():
    prefix = sys.argv[1]
    maps = [m for m in (load_norm(prefix + s) for s in (".ink.jpg", ".ink2d.jpg", ".ink50.jpg")) if m is not None]
    if len(maps) < 2:
        print(f"consensus: <2 maps for {prefix}, skipping")
        return
    h = max(m.shape[0] for m in maps)
    w = max(m.shape[1] for m in maps)
    up = [np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
          for m in maps]
    eps = 1e-3
    logsum = sum(np.log(m + eps) for m in up)
    cons = np.exp(logsum / len(up)) - eps
    Image.fromarray((np.clip(cons, 0, 1) * 255).astype(np.uint8)).save(prefix + ".cons.jpg", quality=92)
    print(f"consensus: {prefix}.cons.jpg from {len(up)} models")


if __name__ == "__main__":
    main()
