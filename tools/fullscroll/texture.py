"""texture — harvest real-CT statistics for phantom realism.

From staged CT + a wind-labeled LabelStore: (a) through-sheet intensity
profiles (CT sampled along the label normal over fdist in [-0.75, 0.75] wraps),
(b) papyrus high-frequency residual patches, (c) air/background patches,
(d) global stats incl. median pitch. One texture_bank.npz consumed only by
phantom.make_phantom(bank=...).
"""
import argparse
import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

BANK_VERSION = 1


class TextureBank:
    def __init__(self, profiles, pap_patches, air_patches, stats):
        self.profiles = profiles
        self.pap = pap_patches
        self.air = air_patches
        self.stats = stats

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta.get("version") != BANK_VERSION:
            raise ValueError(f"texture bank version {meta.get('version')}")
        return TextureBank(z["profiles"], z["pap"], z["air"], meta["stats"])

    def save(self, path, provenance=None):
        meta = {"version": BANK_VERSION, "stats": self.stats,
                "provenance": provenance or {}}
        tmp = path + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, profiles=self.profiles.astype(np.float32),
                            pap=self.pap.astype(np.float16),
                            air=self.air.astype(np.float16),
                            meta=json.dumps(meta))
        os.replace(tmp, path)

    def sample_profile(self, rng):
        return self.profiles[rng.integers(len(self.profiles))]

    def sample_pap_residual(self, rng):
        return np.asarray(self.pap[rng.integers(len(self.pap))], np.float32)

    def sample_air(self, rng):
        return np.asarray(self.air[rng.integers(len(self.air))], np.float32)


def harvest(ct_path, store_path, out_npz, *, n_profiles=20000, n_pap_patches=2000,
            n_air_patches=500, patch=32, profile_halflen_wraps=0.75, Ls=64,
            seed=0, max_tries_factor=50):
    from ctio import open_u8
    from labelstore import LabelStore
    rng = np.random.default_rng([seed])
    store = LabelStore.open(store_path)
    ct = open_u8(ct_path)
    D, H, Wd = store.shape

    def read_ct(o, s):
        return np.asarray(ct[tuple(slice(int(o[i]), int(o[i] + s[i]))
                                   for i in range(3))].read().result(), np.float32)

    # global pass on a coarse grid for pitch + means
    conf = store.read_block("w_conf", (0, 0, 0), store.shape)[::4, ::4, ::4]
    hasw = conf > 0.8
    gz = store.read_block("normal_z", (0, 0, 0), store.shape)[::4, ::4, ::4]
    # pitch from |grad W|: recompute from w channels on a coarse grid
    ws = store.read_block("w_sin", (0, 0, 0), store.shape)[::4, ::4, ::4]
    wc = store.read_block("w_cos", (0, 0, 0), store.shape)[::4, ::4, ::4]
    w = (np.arctan2(ws, wc) / (2 * np.pi)) % 1.0
    from unwrap import wrapdiff
    g = np.stack([wrapdiff(np.diff(w, axis=a, append=w.take([-1], axis=a)))
                  for a in range(3)])
    gn = np.linalg.norm(g, axis=0) / 4.0            # coarse-grid step = 4 vox
    ok = hasw & (gn > 1e-4) & (gn < 0.5)
    pitch_med = float(np.median(1.0 / gn[ok])) if ok.any() else 14.0

    sem_ok = store.has("sem_papyrus")
    pap_c = store.read_block("sem_papyrus", (0, 0, 0), store.shape)[::4, ::4, ::4] \
        if sem_ok else (conf * 0 + 0.6)
    ct_c = read_ct((0, 0, 0), store.shape)[::4, ::4, ::4]
    air_mean = float(ct_c[pap_c < 0.05].mean()) if (pap_c < 0.05).any() else 30.0
    air_std = float(ct_c[pap_c < 0.05].std()) if (pap_c < 0.05).any() else 8.0
    pap_mean = float(ct_c[pap_c > 0.6].mean()) if (pap_c > 0.6).any() else 140.0
    pap_std = float(ct_c[pap_c > 0.6].std()) if (pap_c > 0.6).any() else 20.0

    # profiles
    cand = np.argwhere(hasw & (pap_c > 0.5)) * 4
    if len(cand) < n_profiles // 10:
        raise RuntimeError(f"only {len(cand)} candidate profile voxels")
    profs = []
    tries = 0
    ctf = read_ct((0, 0, 0), store.shape)
    nz = store.read_block("normal_z", (0, 0, 0), store.shape)
    ny = store.read_block("normal_y", (0, 0, 0), store.shape)
    nx = store.read_block("normal_x", (0, 0, 0), store.shape)
    while len(profs) < n_profiles and tries < n_profiles * max_tries_factor:
        tries += 1
        p = cand[rng.integers(len(cand))].astype(np.float64)
        p += rng.uniform(-2, 2, 3)
        n = np.asarray([nz[tuple(p.astype(int))], ny[tuple(p.astype(int))],
                        nx[tuple(p.astype(int))]])
        ln = np.linalg.norm(n)
        if ln < 0.3:
            continue
        n /= ln
        span = profile_halflen_wraps * pitch_med
        ts = np.linspace(-span, span, Ls)
        pts = p[None, :] + ts[:, None] * n[None, :]
        if (pts < 1).any() or (pts > np.asarray([D, H, Wd]) - 2).any():
            continue
        prof = map_coordinates(ctf, pts.T, order=1)
        if prof[Ls // 2 - Ls // 10: Ls // 2 + Ls // 10].mean() < air_mean + 2 * air_std:
            continue                                      # hole hit
        profs.append(prof.astype(np.float32))
    if len(profs) < n_profiles // 2:
        raise RuntimeError(f"harvested only {len(profs)}/{n_profiles} profiles")

    def crops(pred, n_want, tag):
        out = []
        t = 0
        while len(out) < n_want and t < n_want * max_tries_factor:
            t += 1
            o = [rng.integers(0, store.shape[i] - patch) for i in range(3)]
            if not pred(o):
                continue
            c = read_ct(o, (patch,) * 3)
            if (c == 0).mean() > 0.5:
                continue                                  # masked-out region
            out.append(c)
        if len(out) < n_want // 2:
            raise RuntimeError(f"harvested only {len(out)}/{n_want} {tag} patches")
        return np.stack(out)

    def frac_pap(o):
        s = store.read_block("sem_papyrus", o, (patch,) * 3) if sem_ok else None
        return None if s is None else float((s > 0.5).mean())

    pap_patches = crops(lambda o: (fp := frac_pap(o)) is not None and fp > 0.98,
                        n_pap_patches, "papyrus")
    pap_res = pap_patches - np.stack([gaussian_filter(p, 4.0) for p in pap_patches])
    air_patches = crops(lambda o: (fp := frac_pap(o)) is not None and fp < 0.02,
                        n_air_patches, "air")

    stats = {"pap_mean": pap_mean, "pap_std": pap_std, "air_mean": air_mean,
             "air_std": air_std, "pitch_vox_median": pitch_med,
             "n_profiles": len(profs), "profile_halflen_wraps": profile_halflen_wraps}
    bank = TextureBank(np.stack(profs), pap_res, air_patches, stats)
    bank.save(out_npz, provenance={"ct": ct_path, "labels": store_path})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-profiles", type=int, default=20000)
    ap.add_argument("--n-pap", type=int, default=2000)
    ap.add_argument("--n-air", type=int, default=500)
    args = ap.parse_args()
    stats = harvest(args.ct, args.labels, args.out, n_profiles=args.n_profiles,
                    n_pap_patches=args.n_pap, n_air_patches=args.n_air)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
