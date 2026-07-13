"""Winding-identity test for a DISAGREE mesh pair: duplicate tracing or adjacent wraps?

surf-consist's DISAGREE only says the meshes conflict at 2-3 vox in their overlap — on
Paris4 that meant redundant versions of the SAME wrap (dedup correct), on the 0500P2
Khartes corpus it meant DISTINCT adjacent wraps kissing (dedup loses GT), and the local
metrics (overlap/coincident/side-mix) don't separate the cases. The GLOBAL test does:
unwrap both meshes about the umbilicus and compare radius at matched (z, theta) —
same wrap -> |dr| ~ 0 everywhere it matters; adjacent wraps -> |dr| ~ one pitch.

Usage: winding_sep.py --a <tifxyz-dir> --b <tifxyz-dir> --axis <umbilicus.toml>
       (fxsurf inputs: fenix export-tifxyz them first; export_acceptset does this)
Exit prints 'duplicate' / 'distinct' / 'ambiguous' + the dr statistics.

ASYMMETRIC TRUST (measured 2026-07-11): true duplicates read dr ~ floor under ANY axis
(identical geometry maps identically), so a fake 'distinct' is impossible — but a bad
axis (in-bin r smear ~ pitch, e.g. 0500P2's 3-control-point fit at 0.36 vox/z tilt)
makes DISTINCT wraps read 'duplicate'. Consumers may therefore use 'distinct' to rescue
a pair from dedup, and must never use 'duplicate' to force one (default-dedup + manual
--no-dedup covers that side).
"""
import os, sys, argparse, json
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orientation_check import load_axis


def load_dir(d):
    X = tifffile.imread(f"{d}/x.tif").astype(np.float64)
    Y = tifffile.imread(f"{d}/y.tif").astype(np.float64)
    Z = tifffile.imread(f"{d}/z.tif").astype(np.float64)
    V = (X >= 0) & (Y >= 0) & (Z >= 0)
    return X[V], Y[V], Z[V]


def cyl(x, y, z, center_fn):
    cy, cx = center_fn(z)
    dy, dx = y - cy, x - cx
    return np.sqrt(dy * dy + dx * dx), np.arctan2(dy, dx)


def _binned(d, center_fn, zbin, tbins, k, rng):
    """-> {bin_key: sorted radii}. A multi-wrap mesh has MANY radii per (z,theta) bin
    (one per winding crossing that angle) — keep them all; a per-bin median is
    meaningless there (measured: the w010-027 duplicate pair read |dr|~460 on medians)."""
    x, y, z = load_dir(d)
    if len(x) > k:
        s = rng.choice(len(x), k, replace=False)
        x, y, z = x[s], y[s], z[s]
    r, t = cyl(x, y, z, center_fn)
    zi = np.floor(z / zbin).astype(np.int64)
    ti = np.floor((t + np.pi) / (2 * np.pi) * tbins).astype(np.int64) % tbins
    key = zi * tbins + ti
    order = np.argsort(key)
    key, r = key[order], r[order]
    out = {}
    uk, cuts = np.unique(key, return_index=True)
    for i, kk in enumerate(uk):
        hi = cuts[i + 1] if i + 1 < len(uk) else len(r)
        out[int(kk)] = np.sort(r[cuts[i]:hi])
    return out


def binned_self(d, center_fn, zbin, tbins, k, rng):
    """Two INDEPENDENT views of the same mesh — their min-|dr| is the noise floor this
    axis + binning supports. Small meshes (<= k points) must be SPLIT into disjoint
    halves: two full copies are identical and read a fake floor of 0."""
    x, y, z = load_dir(d)
    if len(x) > 2 * k:
        return (_binned(d, center_fn, zbin, tbins, k, rng),
                _binned(d, center_fn, zbin, tbins, k, rng))
    perm = rng.permutation(len(x))
    halves = []
    for idx in (perm[: len(x) // 2], perm[len(x) // 2:]):
        r, t = cyl(x[idx], y[idx], z[idx], center_fn)
        zi = np.floor(z[idx] / zbin).astype(np.int64)
        ti = np.floor((t + np.pi) / (2 * np.pi) * tbins).astype(np.int64) % tbins
        key = zi * tbins + ti
        order = np.argsort(key)
        key, r = key[order], r[order]
        out = {}
        uk, cuts = np.unique(key, return_index=True)
        for i, kk in enumerate(uk):
            hi = cuts[i + 1] if i + 1 < len(uk) else len(r)
            out[int(kk)] = np.sort(r[cuts[i]:hi])
        halves.append(out)
    return halves[0], halves[1]


def winding_sep(dir_a, dir_b, axis_toml, zbin=64.0, tbins=60, k=3_000_000, seed=0):
    # Bin/sample sizing (measured, 2026-07-11): nearest-radius matching needs SAME-WRAP
    # points on both sides of most (z,theta,wrap) granules — at ~1 sample/granule the
    # match lands on the wrong wrap and even self-vs-self reads ~pitch (24.8). 3M samples
    # + (64 z x 6 deg) bins keep granules populated while bounding in-bin r smear from
    # axis drift (Paris4 slope ~0.05 vox/z -> 3 vox over 64 z; a steeply tilted axis
    # still smears — that's what the self-calibrated floor below catches).
    """-> dict with matched-bin |dr| stats. Bins (z, theta); compares median r per bin."""
    center_fn = load_axis(axis_toml)
    rng = np.random.default_rng(seed)

    ma, mb = _binned(dir_a, center_fn, zbin, tbins, k, rng), _binned(dir_b, center_fn, zbin, tbins, k, rng)
    common = sorted(set(ma) & set(mb))
    if len(common) < 10:
        return {"verdict": "ambiguous", "n_bins": len(common), "reason": "too few matched (z,theta) bins"}
    # nearest-radius match: for each A radius, the closest B radius in the same bin.
    # Duplicate/redundant coverage -> min|dr| ~ 0 broadly; adjacent wraps -> ~ one pitch.
    dr = []
    for kk in common:
        ra, rb = ma[kk], mb[kk]
        j = np.clip(np.searchsorted(rb, ra), 0, len(rb) - 1)
        jm = np.clip(j - 1, 0, len(rb) - 1)
        dr.append(np.minimum(np.abs(rb[j] - ra), np.abs(rb[jm] - ra)))
    dr = np.concatenate(dr)
    stats = {"n_bins": len(common), "n_pts": int(len(dr)), "dr_med": float(np.median(dr)),
             "dr_abs_med": float(np.median(np.abs(dr))), "dr_p90": float(np.percentile(np.abs(dr), 90))}
    # SELF-CALIBRATED floor: A-vs-A with an independent subsample measures the noise this
    # axis supports (a tilted/off-center axis smears a single wrap's r across the bin —
    # measured 0500P2: drift 735 vox over a 2k z-span makes wraps INTERLEAVE in-bin, and
    # no matching strategy can discriminate). If the floor is not clearly below the pair's
    # dr, the axis can't answer the question — say so instead of guessing.
    fa, fb = binned_self(dir_a, center_fn, zbin, tbins, k, rng)
    floor = []
    for kk in sorted(set(fa) & set(fb)):
        ra, rb = fa[kk], fb[kk]
        j = np.clip(np.searchsorted(rb, ra), 0, len(rb) - 1)
        jm = np.clip(j - 1, 0, len(rb) - 1)
        floor.append(np.minimum(np.abs(rb[j] - ra), np.abs(rb[jm] - ra)))
    stats["floor_med"] = float(np.median(np.concatenate(floor))) if floor else None
    f = stats["floor_med"] if stats["floor_med"] is not None else 99.0
    d = stats["dr_abs_med"]
    # verdicts are RELATIVE to the floor: the axis only answers questions coarser than
    # its own smear. floor > ~half a pitch -> it can't separate wraps at all.
    if f > 6.0:
        stats["verdict"] = "ambiguous"
        stats["reason"] = "axis-limited: self-floor >= half-pitch scale"
    elif d <= max(4.0, 2.5 * f):
        stats["verdict"] = "duplicate"
    elif d >= max(8.0, 4.0 * f):
        stats["verdict"] = "distinct"
    else:
        stats["verdict"] = "ambiguous"
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--axis", required=True)
    args = ap.parse_args()
    s = winding_sep(args.a, args.b, args.axis)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
