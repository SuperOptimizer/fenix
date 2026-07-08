"""feeder — mixed real+phantom patch feeder for FullScrollNet training.

IterableDataset yielding samples matching losses.full_scroll_loss exactly
(sem/normal/w/ink/interior + wmask/semmask/inkmask/intmask as CONTINUOUS
confidence weights, pitch_vox per sample). Phantoms are generated ON THE FLY in
worker processes (fixed pools measurably overfit the geometry heads); real
crops come from LabelStore-coverage-guided rejection sampling over staged CT.

Determinism: sample i of epoch e is identical regardless of worker count —
per-sample rng seed [seed, epoch, i]; workers stride the epoch index space.
"""
import argparse
import hashlib
import json
import os
import tomllib
from typing import Iterator

import numpy as np
import torch

from labelstore import GROUPS, LabelStore

_HANDLES = {}   # per-process lazy tensorstore handles (never pickled)


def _ct(path):
    if path not in _HANDLES:
        from ctio import open_u8
        _HANDLES[path] = open_u8(path)
    return _HANDLES[path]


def _augment(sample, rng):
    """Flips (all axes) + yx rot90. Spatial index ops on every volume; normal
    transforms as grad W: flip axis a => negate component a; yx rot90 rotates
    the (ny, nx) sub-vector. Volumes are (C,D,H,W); normal channels = (nz,ny,nx)
    ZYX components (pinned contract)."""
    flips = [rng.random() < 0.5 for _ in range(3)]
    k = int(rng.integers(0, 4))
    for key, a in sample.items():
        if not (isinstance(a, np.ndarray) and a.ndim == 4):
            continue
        for ax, f in enumerate(flips):
            if f:
                a = np.flip(a, axis=1 + ax)
        a = np.rot90(a, k, axes=(2, 3))
        sample[key] = np.ascontiguousarray(a)
    n = sample["normal"]
    for ax, f in enumerate(flips):
        if f:
            n[ax] = -n[ax]
    for _ in range(k):     # rot90 in (H,W)=(y,x) plane: (ny,nx) -> (nx,-ny)
        ny, nx = n[1].copy(), n[2].copy()
        n[1], n[2] = nx, -ny
    return sample


class FullScrollDataset(torch.utils.data.IterableDataset):
    def __init__(self, manifest_path, epoch=0):
        super().__init__()
        with open(manifest_path, "rb") as f:
            self.cfg = tomllib.load(f)
        g = self.cfg["global"]
        self.patch = int(g.get("patch", 128))
        self.phantom_frac = float(g.get("phantom_frac", 0.5))
        self.seed = int(g.get("seed", 1234))
        self.epoch_len = int(g.get("epoch_len", 4096))
        self.min_label_frac = float(g.get("min_label_frac", 0.05))
        self.aug = bool(g.get("aug_flips", True))
        self.jitter = bool(g.get("intensity_jitter", True))
        self.epoch = epoch
        self.regions = []
        for r in self.cfg.get("region", []):
            if not (os.path.exists(r["ct"]) and os.path.exists(r["labels"])):
                raise FileNotFoundError(f"region {r['name']}: {r['ct']} / {r['labels']}")
            store = LabelStore.open(r["labels"])
            cands = {}
            for grp in GROUPS:
                try:
                    cov = store.coverage(grp)
                except (FileNotFoundError, KeyError):
                    continue
                idx = np.argwhere(cov >= self.min_label_frac)
                if len(idx):
                    cands[grp] = idx
            if cands:
                self.regions.append({"name": r["name"], "ct": r["ct"],
                                     "labels": r["labels"],
                                     "weight": float(r.get("weight", 1.0)),
                                     "cands": cands, "manifest": store.m})
        if self.phantom_frac < 1.0 and not self.regions:
            raise RuntimeError("no usable regions and phantom_frac < 1")
        ph = self.cfg.get("phantom", {})
        self.bank_path = ph.get("bank", "") or None
        self.pitch_jitter = ph.get("pitch_jitter", [0.75, 1.35])
        self.thick_frac = ph.get("thickness_frac", [0.28, 0.45])
        self._bank = None

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.epoch_len

    def _bank_get(self):
        if self.bank_path and self._bank is None:
            from texture import TextureBank
            self._bank = TextureBank.load(self.bank_path)
        return self._bank

    def _phantom(self, rng):
        from phantom import make_phantom
        bank = self._bank_get()
        base_pitch = (bank.stats["pitch_vox_median"] if bank else 14.0)
        pitch = base_pitch * rng.uniform(*self.pitch_jitter)
        thick = pitch * rng.uniform(*self.thick_frac)
        ph = make_phantom(self.patch, pitch=pitch, thickness=thick,
                          seed=int(rng.integers(2 ** 31)), bank=bank)
        s = {
            "ct": ph["ct"][None],
            "sem": np.stack([ph["papyrus"], ph["recto"], ph["verso"]]).astype(np.float32),
            "normal": ph["normal"].astype(np.float32),
            "w": ph["w"][None].astype(np.float32),
            "ink": ph["ink"][None].astype(np.float32),
            "interior": ph["wmask"][None].astype(np.float32),
            "wmask": ph["wmask"][None].astype(np.float32),
            "semmask": np.ones((1,) + ph["ct"].shape, np.float32),
            "inkmask": np.ones((1,) + ph["ct"].shape, np.float32),
            "intmask": np.ones((1,) + ph["ct"].shape, np.float32),
            "pitch_vox": np.float32(pitch),
        }
        return s

    def _real(self, rng):
        P = self.patch
        weights = np.asarray([r["weight"] * sum(len(v) for v in r["cands"].values())
                              for r in self.regions], np.float64)
        reg = self.regions[rng.choice(len(self.regions), p=weights / weights.sum())]
        store = LabelStore.open(reg["labels"])
        shape = store.shape
        grp = list(reg["cands"])[int(rng.integers(len(reg["cands"])))]
        cands = reg["cands"][grp]
        origin = None
        for _ in range(8):
            c = cands[rng.integers(len(cands))] * np.asarray(store.chunks)
            o = c + rng.integers(-P + 8, np.asarray(store.chunks) - 8, 3)
            o = np.clip(o, 0, np.maximum(np.asarray(shape) - P, 0))
            conf_ch = GROUPS[grp]
            conf = store.read_block(conf_ch, o, (P,) * 3)
            if conf.mean() >= self.min_label_frac:
                origin = o
                break
        if origin is None:
            origin = np.clip(cands[rng.integers(len(cands))]
                             * np.asarray(store.chunks),
                             0, np.maximum(np.asarray(shape) - P, 0))
        o = tuple(int(v) for v in origin)

        def rd(ch):
            return store.read_block(ch, o, (P,) * 3) if store.has(ch) \
                else np.zeros((P,) * 3, np.float32)

        ct = np.asarray(_ct(reg["ct"])[tuple(slice(o[i], o[i] + P)
                                             for i in range(3))].read().result(),
                        np.float32)
        if self.jitter:
            lo, hi = ct.min(), max(float(ct.max()), 1.0)
            ctn = (ct - lo) / max(hi - lo, 1e-6)
            ct = lo + (ctn ** rng.uniform(0.9, 1.1)) * (hi - lo)
            ct = ct + rng.normal(0, 0.02 * (hi - lo), ct.shape).astype(np.float32)
        w_sin, w_cos, w_conf = rd("w_sin"), rd("w_cos"), rd("w_conf")
        w = (np.arctan2(w_sin, w_cos) / (2 * np.pi)) % 1.0
        nrm = np.stack([rd("normal_z"), rd("normal_y"), rd("normal_x")])
        ln = np.linalg.norm(nrm, axis=0)
        wmask = np.where(ln < 0.3, 0.0, w_conf).astype(np.float32)
        nrm = nrm / np.maximum(ln, 1e-6)
        pitch = reg["manifest"].get("provenance", {}).get("pitch_vox_median")
        s = {
            "ct": ct[None],
            "sem": np.stack([rd("sem_papyrus"), rd("sem_recto"), rd("sem_verso")]),
            "normal": nrm.astype(np.float32),
            "w": w[None].astype(np.float32),
            "ink": rd("ink")[None],
            "interior": (w_conf > 0.05).astype(np.float32)[None],
            "wmask": wmask[None],
            "semmask": rd("sem_conf")[None],
            "inkmask": rd("ink_conf")[None],
            # interior negatives only exist on phantoms; real intmask = w cover
            "intmask": (w_conf > 0.0).astype(np.float32)[None],
            "pitch_vox": np.float32(pitch if pitch else np.nan),
        }
        return s

    def __iter__(self) -> Iterator[dict]:
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nworkers = info.num_workers if info else 1
        for i in range(wid, self.epoch_len, nworkers):
            rng = np.random.default_rng([self.seed, self.epoch, i])
            s = self._phantom(rng) if (rng.random() < self.phantom_frac
                                       or not self.regions) else self._real(rng)
            if self.aug:
                s = _augment(s, rng)
            yield s


def collate(samples, device="cuda"):
    x = torch.stack([torch.from_numpy(s["ct"]) for s in samples]).to(device)
    x = (x - x.mean(dim=(1, 2, 3, 4), keepdim=True)) / \
        x.std(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-6)
    t = {}
    for k in ("sem", "normal", "ink", "interior", "wmask", "semmask",
              "inkmask", "intmask"):
        t[k] = torch.stack([torch.from_numpy(s[k]) for s in samples]).to(device)
    t["w"] = torch.stack([torch.from_numpy(s["w"][0]) for s in samples]).to(device)
    t["pitch_vox"] = torch.tensor([float(s["pitch_vox"]) for s in samples],
                                  device=device)
    return x, t


def _collate_cpu(samples):
    return collate(samples, device="cpu")


def make_loader(manifest_path, batch=2, workers=8, epoch=0):
    ds = FullScrollDataset(manifest_path, epoch=epoch)
    return torch.utils.data.DataLoader(
        ds, batch_size=batch, num_workers=workers, collate_fn=_collate_cpu,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        multiprocessing_context="spawn" if workers > 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    import time
    dl = make_loader(args.manifest, args.batch, args.workers)
    t0 = time.time()
    fracs = {}
    first_hash = None
    n = 0
    for x, t in dl:
        if first_hash is None:
            first_hash = hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16]
        for k in ("wmask", "semmask", "inkmask"):
            fracs[k] = fracs.get(k, 0) + float(t[k].mean())
        n += 1
        if n >= args.steps:
            break
    dt = time.time() - t0
    print(f"{n * args.batch / dt:.2f} samples/s | first-batch sha {first_hash} | "
          + " ".join(f"{k}={v/n:.3f}" for k, v in fracs.items()))


if __name__ == "__main__":
    main()
