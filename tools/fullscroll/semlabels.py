"""semlabels — papyrus/recto/verso tolerance-BAND labels with disagreement conf.

Sources: (a) mesh constraint splats (GP meshes are recto surfaces): soft bands
along the point normal, verso displaced by -t*n, papyrus filled between;
(b) teacher prediction zarr (existing surface model output) at fixed moderate
confidence. Sources write scratch channels (sem_recto__srcN ...); reconcile()
merges by confidence-weighted mean and ZEROES confidence where confident
sources disagree (design principle 2: bands + weights, never one-voxel truth).
"""
import argparse
import glob
import json
import os

import numpy as np

from labelstore import LabelStore


def _ramp(d, lo, hi):
    return np.clip((hi - d) / max(hi - lo, 1e-6), 0.0, 1.0)


def render_mesh_bands(store, constraint_files, tol_vox=2.0, thickness=5.0,
                      src_tag="src0", base_conf=0.9):
    """Splat soft recto/verso/papyrus bands from constraint points into
    scratch channels sem_*__<src_tag>."""
    D, H, Wd = store.shape
    stamp_r = int(np.ceil(tol_vox + thickness + 2))
    acc = {k: np.zeros((D, H, Wd), np.float32)
           for k in ("recto", "verso", "papyrus")}
    hit = np.zeros((D, H, Wd), np.float32)
    off = np.stack(np.meshgrid(*([np.arange(-stamp_r, stamp_r + 1)] * 3),
                               indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
    for f in constraint_files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        q = float(meta.get("mean_radial_dot", 0.5))
        pos, nrm = z["pos_zyx"], z["normal_zyx"]
        step = max(1, len(pos) // 200000)
        for p, n in zip(pos[::step], nrm[::step]):
            c = np.round(p).astype(np.int64)
            if not ((c >= -stamp_r) & (c < np.asarray([D, H, Wd]) + stamp_r)).all():
                continue
            vox = c + off
            ok = ((vox >= 0) & (vox < np.asarray([D, H, Wd]))).all(1)
            vox = vox[ok].astype(np.int64)
            d = ((vox - p) * n).sum(1)                    # signed dist along normal
            perp = np.linalg.norm((vox - p) - d[:, None] * n, axis=1)
            near = perp < tol_vox + 1.5
            vox, d = vox[near], d[near]
            idx = tuple(vox.T)
            np.maximum.at(acc["recto"], idx, _ramp(np.abs(d), tol_vox - 1, tol_vox + 1))
            np.maximum.at(acc["verso"], idx,
                          _ramp(np.abs(d + thickness), tol_vox - 1, tol_vox + 1))
            pap = np.where(d <= 0, _ramp(np.abs(d + thickness / 2),
                                         thickness / 2, thickness / 2 + 2), 0.0)
            np.maximum.at(acc["papyrus"], idx, pap)
            np.maximum.at(hit, idx, q * base_conf)
    for name in ("papyrus", "recto", "verso"):
        _write_scratch(store, f"sem_{name}__{src_tag}", acc[name])
    _write_scratch(store, f"sem_conf__{src_tag}", hit)


def ingest_teacher_zarr(store, pred_zarr, channel_map=None, conf=0.6,
                        src_tag="src1"):
    """Teacher zarr (u8 prob [Z,Y,X], region-aligned) -> scratch channels."""
    from ctio import open_u8
    channel_map = channel_map or {"sem_recto": ""}
    h = open_u8(pred_zarr)
    D, H, Wd = store.shape
    data = np.asarray(h[:D, :H, :Wd].read().result(), np.float32) / 255.0
    for ch in channel_map:
        _write_scratch(store, f"{ch}__{src_tag}", data)
    _write_scratch(store, f"sem_conf__{src_tag}",
                   np.full((D, H, Wd), conf, np.float32) * (data > 0.02))


def _write_scratch(store, channel, arr):
    if not store.has(channel):
        from numcodecs import Blosc
        from labelstore import CHANNELS, FILL
        kind = CHANNELS[channel.split("__", 1)[0]]
        store.g.create_array(channel, shape=store.shape, chunks=store.chunks,
                             dtype="u1", fill_value=FILL[kind],
                             compressors=Blosc(cname="zstd", clevel=5,
                                               shuffle=Blosc.BITSHUFFLE),
                             chunk_key_encoding={"name": "v2", "separator": "/"})
        store.m["channels"][channel] = kind
        LabelStore._write_manifest(store.path, store.m)
    for z0 in range(0, store.shape[0], store.chunks[0]):
        s = slice(z0, min(z0 + store.chunks[0], store.shape[0]))
        store.write_block(channel, (z0, 0, 0), arr[s])


def reconcile(store, disagree_thresh=0.5, sigma_d=0.25):
    """Merge sem_*__src* scratch -> sem_* + sem_conf; conf=0 where >=2 confident
    sources disagree by > disagree_thresh. Deletes scratch channels."""
    tags = sorted({c.split("__", 1)[1] for c in store.m["channels"] if "__" in c})
    D = store.shape
    for z0 in range(0, D[0], store.chunks[0]):
        s = slice(z0, min(z0 + store.chunks[0], D[0]))
        shape = (s.stop - s.start, D[1], D[2])
        confs = [store.read_block(f"sem_conf__{t}", (z0, 0, 0), shape)
                 for t in tags]
        merged_conf = np.zeros(shape, np.float32)
        for name in ("papyrus", "recto", "verso"):
            vals = [store.read_block(f"sem_{name}__{t}", (z0, 0, 0), shape)
                    if store.has(f"sem_{name}__{t}") else None for t in tags]
            num = np.zeros(shape, np.float32)
            den = np.zeros(shape, np.float32)
            vmin = np.full(shape, np.inf, np.float32)
            vmax = np.full(shape, -np.inf, np.float32)
            nconf = np.zeros(shape, np.int16)
            for v, c in zip(vals, confs):
                if v is None:
                    continue
                num += v * c
                den += c
                conf_here = c > 0.3
                vmin = np.where(conf_here, np.minimum(vmin, v), vmin)
                vmax = np.where(conf_here, np.maximum(vmax, v), vmax)
                nconf += conf_here
            val = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
            spread = np.where(nconf >= 2, vmax - vmin, 0.0)
            conf = np.where(den > 0,
                            np.clip(den / len(tags), 0, 1) * np.exp(-spread / sigma_d),
                            0.0)
            conf = np.where(spread > disagree_thresh, 0.0, conf)
            store.write_block(f"sem_{name}", (z0, 0, 0), val.astype(np.float32))
            merged_conf = np.maximum(merged_conf, conf.astype(np.float32))
        store.write_block("sem_conf", (z0, 0, 0), merged_conf)
    for c in [c for c in list(store.m["channels"]) if "__" in c]:
        store.delete_channel(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True)
    ap.add_argument("--mesh-constraints", default="")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--thickness", type=float, default=5.0)
    args = ap.parse_args()
    store = LabelStore.open(args.store)
    # ensure output channels exist
    for name in ("sem_papyrus", "sem_recto", "sem_verso", "sem_conf"):
        if not store.has(name):
            _write_scratch(store, name, np.zeros(store.shape, np.float32))
    if args.mesh_constraints:
        render_mesh_bands(store, sorted(glob.glob(args.mesh_constraints)),
                          tol_vox=args.tol, thickness=args.thickness)
    if args.teacher:
        ingest_teacher_zarr(store, args.teacher)
    reconcile(store)
    store.finalize()
    print("semlabels done:", store.m["coverage"].get("sem"))


if __name__ == "__main__":
    main()
