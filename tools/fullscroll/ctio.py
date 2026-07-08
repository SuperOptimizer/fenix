"""CT access, block/halo iteration, staging, and the prediction-store writer.

Read path: tensorstore over LOCAL staged zarr v2 u8 [Z,Y,X]. Staging from S3/http
is `fenix ingest-zarr`'s job (battle-tested retry-vs-absent semantics) — Python
never does remote CT IO. Doctrine: tensorstore returns fill for absent chunks,
which is correct for masked scrolls (absent = air) ONLY because the store is
local and complete; that is why the feeder/sweep read local NVMe only.

Normalization is pinned HERE, once: per-patch z-score (mean/std of the patch),
identical to training (`phantom.to_batch`) and `load_ct_patch`. Any seam
artifacts trace to this choice; do not renormalize anywhere else.
"""
import json
import os
import subprocess
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np

PRED_VERSION = 1


def open_u8(path, cache_bytes=2 << 30):
    import tensorstore as ts
    spec = {
        "driver": "zarr",
        "kvstore": {"driver": "file", "path": path},
        "context": {"cache_pool": {"total_bytes_limit": int(cache_bytes)}},
    }
    return ts.open(spec, read=True).result()


def read_patch(handle, origin_zyx, shape_zyx):
    """Returns (data_u8 [D,H,W], inbounds_mask bool [D,H,W]); out-of-bounds = 0."""
    o = np.asarray(origin_zyx, np.int64)
    s = np.asarray(shape_zyx, np.int64)
    vol = np.asarray(handle.shape[-3:], np.int64)
    lo = np.maximum(o, 0)
    hi = np.minimum(o + s, vol)
    out = np.zeros(tuple(s), np.uint8)
    mask = np.zeros(tuple(s), bool)
    if (hi > lo).all():
        src = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3))
        dst = tuple(slice(int(lo[i] - o[i]), int(hi[i] - o[i])) for i in range(3))
        out[dst] = handle[src].read().result()
        mask[dst] = True
    return out, mask


def zscore(x_u8):
    x = np.asarray(x_u8, np.float32)
    return (x - x.mean()) / max(float(x.std()), 1e-6)


@dataclass
class BlockSpec:
    index_zyx: tuple
    core: tuple      # slices, global coords; cores tile the bbox disjointly
    halo: tuple      # core expanded by halo vox, clamped


def iter_blocks(shape_zyx, block=128, halo=16, bbox=None) -> Iterator[BlockSpec]:
    lo = (0, 0, 0) if bbox is None else tuple(int(v) for v in bbox[0])
    hi = tuple(shape_zyx) if bbox is None else tuple(int(v) for v in bbox[1])
    nb = [range(lo[i], hi[i], block) for i in range(3)]
    for bz in nb[0]:
        for by in nb[1]:
            for bx in nb[2]:
                o = (bz, by, bx)
                core = tuple(slice(o[i], min(o[i] + block, hi[i])) for i in range(3))
                hal = tuple(slice(max(core[i].start - halo, lo[i]),
                                  min(core[i].stop + halo, hi[i])) for i in range(3))
                idx = tuple((o[i] - lo[i]) // block for i in range(3))
                yield BlockSpec(idx, core, hal)


def stage(url, level, out_path, origin_zyx, shape_zyx,
          fenix_bin="./build-release/fenix"):
    cmd = [fenix_bin, "ingest-zarr", url, str(level), out_path,
           *[str(int(v)) for v in origin_zyx], *[str(int(v)) for v in shape_zyx]]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"fenix ingest-zarr failed ({r.returncode}): {' '.join(cmd)}")


# ---- prediction store (inference output; solver input) -------------------------
# Same encode/layout family as LabelStore, plus done/culled brick ledger and
# provenance attrs (scroll_id, level, voxel_um, ckpt/config hashes). Channels are
# the net's heads; `interior` doubles as the solver's confidence.
PRED_CHANNELS = {
    "papyrus": "prob", "recto": "prob", "verso": "prob", "interior": "prob",
    "w_sin": "snorm", "w_cos": "snorm",
    "normal_z": "snorm", "normal_y": "snorm", "normal_x": "snorm",
    "ink": "prob",
}


class PredStore:
    """Chunked u8 zarr-v2 group for sweep outputs; chunk grid == solver block grid.

    Resumability: per-brick markers in <path>/ledger/ — "<bz>_<by>_<bx>.done"
    (json with provenance) or ".culled" (distinct, auditable — a culled brick is
    NEVER silently identical to an empty done brick). Resume refuses bricks whose
    marker provenance hash mismatches unless force=True.
    """

    def __init__(self, path, meta, group):
        self.path, self.m, self.g = path, meta, group
        self.chunks = tuple(meta["chunks"])
        self.shape = tuple(meta["shape_zyx"])
        self._ledger = os.path.join(path, "ledger")

    @staticmethod
    def create(path, *, scroll_id, level, voxel_um, shape_zyx, channels=None,
               chunks=(128, 128, 128), provenance=None, overwrite=False):
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(path)
        from numcodecs import Blosc
        import zarr
        from labelstore import FILL
        channels = channels or list(PRED_CHANNELS)
        g = zarr.open_group(path, mode="w", zarr_format=2)
        comp = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
        for c in channels:
            g.create_array(c, shape=tuple(shape_zyx), chunks=tuple(chunks),
                           dtype="u1", fill_value=FILL[PRED_CHANNELS[c]],
                           compressors=comp, chunk_key_encoding={"name": "v2", "separator": "/"},
                           overwrite=overwrite)
        meta = {"pred_version": PRED_VERSION, "scroll_id": scroll_id,
                "level": int(level), "voxel_um": float(voxel_um),
                "shape_zyx": list(shape_zyx), "chunks": list(chunks),
                "channels": channels, "provenance": provenance or {}}
        g.attrs.update(meta)
        os.makedirs(os.path.join(path, "ledger"), exist_ok=True)
        tmp = os.path.join(path, f"meta.json.tmp.{os.getpid()}")
        with open(tmp, "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(tmp, os.path.join(path, "meta.json"))
        return PredStore(path, meta, g)

    @staticmethod
    def open(path):
        import zarr
        with open(os.path.join(path, "meta.json")) as f:
            m = json.load(f)
        if m.get("pred_version") != PRED_VERSION:
            raise ValueError(f"pred_version {m.get('pred_version')} != {PRED_VERSION}")
        return PredStore(path, m, zarr.open_group(path, mode="r+", zarr_format=2))

    def write_brick(self, origin_zyx, fields: dict):
        """fields: channel -> f32 array, all same shape; origin chunk-aligned."""
        o = tuple(int(v) for v in origin_zyx)
        from labelstore import encode_u8
        for c, a in fields.items():
            sl = tuple(slice(o[i], o[i] + a.shape[i]) for i in range(3))
            self.g[c][sl] = encode_u8(a, PRED_CHANNELS[c])

    def read_brick(self, channel, origin_zyx, shape_zyx):
        from labelstore import decode_u8
        sl = tuple(slice(int(origin_zyx[i]), int(origin_zyx[i]) + int(shape_zyx[i]))
                   for i in range(3))
        return decode_u8(self.g[channel][sl], PRED_CHANNELS[channel])

    # -- brick ledger ---------------------------------------------------------
    def _marker(self, index_zyx, kind):
        return os.path.join(self._ledger, "%d_%d_%d.%s" % (*index_zyx, kind))

    def mark(self, index_zyx, kind, info=None):
        assert kind in ("done", "culled")
        tmp = self._marker(index_zyx, kind) + f".tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"provenance": self.m["provenance"], **(info or {})}, f)
        os.replace(tmp, self._marker(index_zyx, kind))

    def status(self, index_zyx):
        for kind in ("done", "culled"):
            if os.path.exists(self._marker(index_zyx, kind)):
                return kind
        return None

    def check_resume(self, index_zyx, force=False):
        """None if brick needs work; 'done'/'culled' to skip. Raises on
        provenance mismatch (older ckpt/config wrote it) unless force."""
        kind = self.status(index_zyx)
        if kind is None:
            return None
        with open(self._marker(index_zyx, kind)) as f:
            info = json.load(f)
        if info.get("provenance") != self.m["provenance"] and not force:
            raise RuntimeError(
                f"brick {index_zyx} was written with different provenance "
                f"{info.get('provenance')} — rerun with --force-restart to accept")
        return kind
