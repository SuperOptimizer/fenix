"""inkproject — ink labels: 2D flattened preds back-projected + 3D teacher ingest.

project_ink_2d: for each valid PPM pixel, splat the 2D ink probability along
-n_hat over a shell (2D ink models sample BELOW the recto surface), max-combined.
Labeled-negative doctrine: within a projected segment's shell, ink_conf is set
even where ink ~ 0 (real negatives); elsewhere ink_conf = 0 (unlabeled is NOT
negative). ingest_ink_3d: voxel-space teacher volume, direct or resampled copy.
Both are noisy teachers -> the trainer treats ink as distillation (soft targets).
"""
import argparse
import json
import os

import numpy as np

from labelstore import LabelStore
from mesh_ingest import load_ppm, xyz_to_zyx


def project_ink_2d(store, ppm_path, ink_png_path, shell=(-8, 2),
                   teacher_conf=0.5, scale=1.0, offset=(0, 0, 0), step=2):
    from PIL import Image
    ink2d = np.asarray(Image.open(ink_png_path).convert("L"), np.float32) / 255.0
    d = load_ppm(ppm_path, step=step)
    val = d["valid"]
    # align ink image to the (subsampled) ppm grid
    ih, iw = ink2d.shape
    ph, pw = val.shape
    ys = (np.arange(ph) * step * ih / (ph * step)).astype(np.int64).clip(0, ih - 1)
    xs = (np.arange(pw) * step * iw / (pw * step)).astype(np.int64).clip(0, iw - 1)
    ink_g = ink2d[np.ix_(ys, xs)]
    pos = xyz_to_zyx(d["pos_xyz"][val]) * scale + np.asarray(offset, np.float64)
    nrm = xyz_to_zyx(d["normal_xyz"][val])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    prob = ink_g[val]
    D, H, Wd = store.shape
    acc = np.zeros((D, H, Wd), np.float32)
    conf = np.zeros((D, H, Wd), np.float32)
    depths = np.arange(shell[0], shell[1] + 1, dtype=np.float32)
    for p, n, v in zip(pos, nrm, prob):
        pts = np.round(p[None, :] + depths[:, None] * n[None, :]).astype(np.int64)
        ok = ((pts >= 0) & (pts < np.asarray([D, H, Wd]))).all(1)
        idx = tuple(pts[ok].T)
        np.maximum.at(acc, idx, v)
        np.maximum.at(conf, idx, teacher_conf)
    _accumulate(store, acc, conf)
    return {"points": int(val.sum()), "shell": list(shell)}


def ingest_ink_3d(store, ink_zarr, conf=0.6):
    from ctio import open_u8
    h = open_u8(ink_zarr)
    D, H, Wd = store.shape
    data = np.asarray(h[:D, :H, :Wd].read().result(), np.float32) / 255.0
    _accumulate(store, data, np.full_like(data, conf) * (data >= 0.0))
    return {"voxels": data.size}


def _accumulate(store, ink_f32, conf_f32):
    for z0 in range(0, store.shape[0], store.chunks[0]):
        s = slice(z0, min(z0 + store.chunks[0], store.shape[0]))
        shape = (s.stop - s.start,) + store.shape[1:]
        cur = store.read_block("ink", (z0, 0, 0), shape)
        curc = store.read_block("ink_conf", (z0, 0, 0), shape)
        store.write_block("ink", (z0, 0, 0), np.maximum(cur, ink_f32[s]))
        store.write_block("ink_conf", (z0, 0, 0), np.maximum(curc, conf_f32[s]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--ppm", default="")
    ap.add_argument("--ink-png", default="")
    ap.add_argument("--ink-zarr", default="")
    ap.add_argument("--shell", type=int, nargs=2, default=[-8, 2])
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--offset", type=float, nargs=3, default=[0, 0, 0])
    args = ap.parse_args()
    store = LabelStore.open(args.store)
    from semlabels import _write_scratch
    for name in ("ink", "ink_conf"):
        if not store.has(name):
            _write_scratch(store, name, np.zeros(store.shape, np.float32))
    if args.ppm and args.ink_png:
        print(project_ink_2d(store, args.ppm, args.ink_png, tuple(args.shell),
                             scale=args.scale, offset=tuple(args.offset)))
    if args.ink_zarr:
        print(ingest_ink_3d(store, args.ink_zarr))
    store.finalize()
    print("ink coverage:", store.m["coverage"].get("ink"))


if __name__ == "__main__":
    main()
