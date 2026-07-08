"""LabelStore — the (value, confidence) dense-label container (labelstore_version 1).

One on-disk contract for every dense label field. Producers: fitfield.py,
semlabels.py, inkproject.py (later: `fenix winding render-labels`). Consumer:
feeder.py (training) and solver/ (w channels of prediction stores share the same
encode). Zarr v2 group, one array per channel, plain [Z,Y,X] u8, 128^3 chunks
aligned to the CT chunk grid at the same level, dimension_separator "/",
blosc-zstd. Single-channel u8 [Z,Y,X] is exactly what `fenix ingest-zarr` reads
and `fenix transcode --scale255` converts to .fxvol — zero new C++ downstream.

Every label is (value, confidence): confidence lives in a *_conf channel and is
the per-voxel loss weight. Missing chunk = decoded fill = confidence 0, never a
value. Coordinates ZYX at a stated (scroll_id, level); readers reject version or
grid mismatches (no silent compat).
"""
import base64
import json
import os

import numpy as np

VERSION = 1

CHANNELS = {
    "w_sin": "snorm",       # sin(2*pi*w)
    "w_cos": "snorm",
    "w_conf": "prob",
    "normal_z": "snorm",    # unit normal ZYX components, toward increasing W
    "normal_y": "snorm",
    "normal_x": "snorm",
    "sem_papyrus": "prob",  # soft targets
    "sem_recto": "prob",
    "sem_verso": "prob",
    "sem_conf": "prob",
    "ink": "prob",
    "ink_conf": "prob",
}
# per-head-group confidence channel (coverage + feeder masks); normal shares w_conf
GROUPS = {"wind": "w_conf", "sem": "sem_conf", "ink": "ink_conf"}
FILL = {"snorm": 128, "prob": 0}


def encode_u8(a, kind):
    a = np.asarray(a, np.float32)
    bad = ~np.isfinite(a)
    if bad.any():
        if kind == "prob":
            raise ValueError(f"NaN/inf in a prob/conf channel ({bad.sum()} voxels)")
        a = np.where(bad, 0.0, a)
    if kind == "snorm":
        return np.clip(np.round(a * 127.5 + 127.5), 0, 255).astype(np.uint8)
    if kind == "prob":
        return np.clip(np.round(a * 255.0), 0, 255).astype(np.uint8)
    raise KeyError(kind)


def decode_u8(a, kind):
    a = np.asarray(a, np.float32)
    if kind == "snorm":
        return (a - 127.5) / 127.5
    if kind == "prob":
        return a / 255.0
    raise KeyError(kind)


def _zarr_open_group(path, mode):
    import zarr
    return zarr.open_group(path, mode=mode, zarr_format=2)


class LabelStore:
    def __init__(self, path, manifest, group):
        self.path = path
        self.m = manifest
        self.g = group
        self.chunks = tuple(manifest["chunks"])
        self.shape = tuple(manifest["shape_zyx"])

    # -- lifecycle ---------------------------------------------------------
    @staticmethod
    def create(path, *, scroll_id, level, voxel_um, origin_zyx, shape_zyx,
               channels, chunks=(128, 128, 128), provenance=None, overwrite=False):
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)
        for c in channels:
            base = c.split("__", 1)[0]
            if base not in CHANNELS:
                raise KeyError(f"unknown channel {c}")
        from numcodecs import Blosc
        g = _zarr_open_group(path, "w")
        comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        for c in channels:
            kind = CHANNELS[c.split("__", 1)[0]]
            g.create_array(c, shape=tuple(shape_zyx), chunks=tuple(chunks),
                           dtype="u1", fill_value=FILL[kind], compressors=comp,
                           chunk_key_encoding={"name": "v2", "separator": "/"}, overwrite=overwrite)
        manifest = {
            "labelstore_version": VERSION, "scroll_id": scroll_id, "level": int(level),
            "voxel_um": float(voxel_um), "origin_zyx": list(origin_zyx),
            "shape_zyx": list(shape_zyx), "chunks": list(chunks),
            "channels": {c: CHANNELS[c.split("__", 1)[0]] for c in channels},
            "provenance": provenance or {}, "coverage": {},
        }
        LabelStore._write_manifest(path, manifest)
        return LabelStore(path, manifest, g)

    @staticmethod
    def open(path):
        mp = os.path.join(path, "manifest.json")
        with open(mp) as f:
            m = json.load(f)
        if m.get("labelstore_version") != VERSION:
            raise ValueError(f"labelstore version {m.get('labelstore_version')} != {VERSION}")
        return LabelStore(path, m, _zarr_open_group(path, "r+"))

    @staticmethod
    def _write_manifest(path, manifest):
        tmp = os.path.join(path, f"manifest.json.tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, os.path.join(path, "manifest.json"))

    # -- IO ------------------------------------------------------------------
    def has(self, channel):
        return channel in self.m["channels"]

    def _kind(self, channel):
        return self.m["channels"][channel]

    def write_block(self, channel, origin_zyx, data_f32):
        o = tuple(int(v) for v in origin_zyx)
        s = data_f32.shape
        for i in range(3):
            if o[i] % self.chunks[i] != 0:
                raise ValueError(f"origin {o} not chunk-aligned ({self.chunks})")
            if o[i] + s[i] > self.shape[i]:
                raise ValueError(f"block {o}+{s} exceeds region {self.shape}")
            if s[i] % self.chunks[i] != 0 and o[i] + s[i] != self.shape[i]:
                raise ValueError(f"block shape {s} not chunk-multiple nor at edge")
        sl = tuple(slice(o[i], o[i] + s[i]) for i in range(3))
        self.g[channel][sl] = encode_u8(data_f32, self._kind(channel))

    def read_block(self, channel, origin_zyx, shape_zyx):
        o, s = origin_zyx, shape_zyx
        sl = tuple(slice(int(o[i]), int(o[i]) + int(s[i])) for i in range(3))
        return decode_u8(self.g[channel][sl], self._kind(channel))

    def delete_channel(self, channel):
        import shutil
        shutil.rmtree(os.path.join(self.path, channel), ignore_errors=True)
        del self.m["channels"][channel]
        LabelStore._write_manifest(self.path, self.m)

    # -- coverage ------------------------------------------------------------
    def finalize(self):
        """Per-chunk labeled fraction per head group, saved to coverage.npz +
        summary in manifest. Uses the group's conf channel; group skipped if the
        conf channel is absent."""
        cov = {}
        nchunks = tuple((self.shape[i] + self.chunks[i] - 1) // self.chunks[i]
                        for i in range(3))
        for grp, conf_ch in GROUPS.items():
            if not self.has(conf_ch):
                continue
            arr = np.zeros(nchunks, np.float32)
            a = self.g[conf_ch]
            for cz in range(nchunks[0]):
                for cy in range(nchunks[1]):
                    for cx in range(nchunks[2]):
                        sl = tuple(slice(c * self.chunks[i],
                                         min((c + 1) * self.chunks[i], self.shape[i]))
                                   for i, c in enumerate((cz, cy, cx)))
                        arr[cz, cy, cx] = float((a[sl] > 0).mean())
            cov[grp] = arr
        np.savez_compressed(os.path.join(self.path, "coverage.npz"), **cov)
        self.m["coverage"] = {g: {"frac_chunks_labeled": float((v > 0).mean()),
                                  "mean_frac": float(v.mean())} for g, v in cov.items()}
        LabelStore._write_manifest(self.path, self.m)

    def coverage(self, group):
        p = os.path.join(self.path, "coverage.npz")
        if not os.path.exists(p):
            raise FileNotFoundError("finalize() not run")
        with np.load(p) as z:
            return z[group]
