"""fitfield — constraints -> dense (w, normal, confidence) label fields (fit v0).

The bootstrap label generator: a cylindrical winding-field fit. Per (z-slab,
theta-bin), monotone radial profiles of W are built from all overlapping mesh
constraints with outlier rejection, then rendered densely over a region
(Eulerian view: air gaps included — papyrus-only w leaves wraps disconnected,
measured 0.15 vs 1.00 wrap-acc). Per-mesh integer wrap offsets K_m are solved
exactly like unwrap.stitch: median pairwise observations -> integer edges ->
spanning tree + loop residuals; conflicting sectors get their confidence cut.

When src/winding's diffeomorphic fit lands, `fenix winding render-labels`
replaces this producer writing the identical LabelStore contract.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from labelstore import LabelStore
from mesh_ingest import load_umbilicus, umbilicus_at


@dataclass
class FitTable:
    z_lo: float
    nz_slab: float          # slab thickness (vox)
    ntheta: int
    nslabs: int
    # nodes[(slab, tbin)] = (r f32 [K] ascending, W f32 [K], conf f32 [K])
    nodes: dict

    def save(self, path):
        keys = np.asarray(list(self.nodes.keys()), np.int64)
        flat_r = np.concatenate([self.nodes[tuple(k)][0] for k in keys]) if len(keys) else np.zeros(0, np.float32)
        flat_W = np.concatenate([self.nodes[tuple(k)][1] for k in keys]) if len(keys) else np.zeros(0, np.float32)
        flat_c = np.concatenate([self.nodes[tuple(k)][2] for k in keys]) if len(keys) else np.zeros(0, np.float32)
        counts = np.asarray([len(self.nodes[tuple(k)][0]) for k in keys], np.int64)
        tmp = path + f".tmp.{os.getpid()}.npz"
        np.savez_compressed(tmp, keys=keys, counts=counts, r=flat_r, W=flat_W,
                            c=flat_c, meta=json.dumps({"z_lo": self.z_lo,
                            "nz_slab": self.nz_slab, "ntheta": self.ntheta,
                            "nslabs": self.nslabs}))
        os.replace(tmp, path)

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        nodes = {}
        off = 0
        for k, n in zip(z["keys"], z["counts"]):
            nodes[tuple(k)] = (z["r"][off:off + n], z["W"][off:off + n],
                               z["c"][off:off + n])
            off += n
        return FitTable(meta["z_lo"], meta["nz_slab"], meta["ntheta"],
                        meta["nslabs"], nodes)


def _load_constraints(files):
    out = []
    for f in files:
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        out.append({"pos": z["pos_zyx"], "theta": z["theta"],
                    "normal": z["normal_zyx"], "weight": z["weight"],
                    "id": meta["mesh_id"] + f"_c{meta.get('component', 0)}",
                    "quality": float(meta.get("mean_radial_dot", 0.5)),
                    "meta": meta})
    return out


def _bin_points(cons, umb, z_lo, nz_slab, ntheta):
    """-> bins[(slab,tbin)] = list of (r, mesh_idx, theta_star)."""
    bins = defaultdict(list)
    for mi, c in enumerate(cons):
        p = c["pos"]
        ctr = umbilicus_at(umb, p[:, 0])
        r = np.hypot(p[:, 1] - ctr[:, 0], p[:, 2] - ctr[:, 1])
        th = np.arctan2(p[:, 1] - ctr[:, 0], p[:, 2] - ctr[:, 1])
        slab = ((p[:, 0] - z_lo) / nz_slab).astype(np.int64)
        tbin = ((th + np.pi) / (2 * np.pi) * ntheta).astype(np.int64) % ntheta
        for i in range(len(p)):
            bins[(int(slab[i]), int(tbin[i]))].append(
                (float(r[i]), mi, float(c["theta"][i])))
    return bins


def solve_mesh_offsets(cons, bins, r_merge=2.0, mad_max=0.35, resid_max=0.3):
    """Per-mesh integer offsets K_m with W = K_m + theta*/(2*pi). Returns
    (offsets list or None per mesh, loop_residuals, conflict_pairs).

    Observations come from SAME-cluster point pairs only (two meshes crossing
    the same wrap in the same sector): obs = -(theta*_b - theta*_a)/2pi = K_b -
    K_a exactly. Cross-cluster (rank-1) terms were tried and measured toxic on
    real GP meshes: cluster split/merge noise shifts them by exactly +-1, giving
    every pair a 50/50 bimodal split (MAD ~ 1.0, zero usable edges) — while
    same-cluster obs have MAD ~ 0.002 with 10^4 samples per pair."""
    pair_obs = defaultdict(list)
    for pts in bins.values():
        pts = sorted(pts)
        clusters = []
        for r, mi, th in pts:
            if clusters and r - clusters[-1][-1][0] < r_merge:
                clusters[-1].append((r, mi, th))
            else:
                clusters.append([(r, mi, th)])
        for cl in clusters:
            for i in range(len(cl)):
                for j in range(i + 1, len(cl)):
                    ra, ma, ta = cl[i]
                    rb, mb, tb = cl[j]
                    if ma == mb:
                        continue
                    obs = -(tb - ta) / (2 * np.pi)
                    key = (min(ma, mb), max(ma, mb))
                    pair_obs[key].append(obs if ma < mb else -obs)
    edges = []
    for (a, b), obs in pair_obs.items():
        obs = np.asarray(obs)
        med = np.median(obs)
        mad = np.median(np.abs(obs - med))
        if mad > mad_max or len(obs) < 3:
            continue
        edges.append((a, b, float(med), len(obs)))
    # spanning tree from the best-covered mesh
    n = len(cons)
    K = [None] * n
    if not edges:
        return K, [], []
    deg = defaultdict(int)
    for a, b, _, w in edges:
        deg[a] += w
        deg[b] += w
    root = max(deg, key=deg.get)
    K[root] = 0
    residuals, conflicts = [], []
    pending = sorted(edges, key=lambda e: -e[3])
    progressed = True
    while progressed:
        progressed = False
        rest = []
        for a, b, d, w in pending:
            if K[a] is not None and K[b] is None:
                K[b] = K[a] + round(d)
                progressed = True
            elif K[b] is not None and K[a] is None:
                K[a] = K[b] - round(d)
                progressed = True
            elif K[a] is not None and K[b] is not None:
                r = (K[a] + d) - K[b]
                residuals.append(float(r))
                # BOTH failure modes are conflicts: fractional residual = noisy
                # geometry; nonzero INTEGER residual = sheet-switch error (a mesh
                # jumps wraps mid-segment — measured on real GP meshes: exact
                # residuals of -10..+10 with 0 fractional part)
                if abs(r - round(r)) > resid_max or abs(round(r)) >= 1:
                    conflicts.append((cons[a]["id"], cons[b]["id"], float(r)))
            else:
                rest.append((a, b, d, w))
        pending = rest
    return K, residuals, conflicts


def build_table(cons, bins, offsets, z_lo, nz_slab, ntheta, conflicts=(),
                w_cluster=0.25, sigma0=2.0, mono_tol=0.25):
    conflict_meshes = {m for pair in conflicts for m in pair[:2]}
    nodes = {}
    max_slab = 0
    for key, pts in bins.items():
        samples = []
        for r, mi, th in pts:
            if offsets[mi] is None:
                continue
            W = offsets[mi] + th / (2 * np.pi)
            pen = 0.3 if cons[mi]["id"] in conflict_meshes else 1.0
            samples.append((W, r, cons[mi]["quality"] * pen))
        if not samples:
            continue
        samples.sort()
        groups = []
        for W, r, q in samples:
            if groups and W - groups[-1][-1][0] < w_cluster:
                groups[-1].append((W, r, q))
            else:
                groups.append([(W, r, q)])
        rs, Ws, cs = [], [], []
        for g in groups:
            W = float(np.mean([x[0] for x in g]))
            rmed = float(np.median([x[1] for x in g]))
            sig = float(np.std([x[1] for x in g]))
            conf = min(1.0, len(g) / 8.0) * np.exp(-sig / sigma0) * \
                float(np.mean([x[2] for x in g]))
            rs.append(rmed)
            Ws.append(W)
            cs.append(conf)
        order = np.argsort(rs)
        rs, Ws, cs = (np.asarray(v, np.float32)[order] for v in (rs, Ws, cs))
        keep = np.ones(len(rs), bool)          # monotone: W increasing with r
        for i in range(1, len(rs)):
            j = np.nonzero(keep[:i])[0]
            if len(j) and Ws[i] < Ws[j[-1]] - mono_tol:
                keep[i] = False
        if keep.sum() == 0:
            continue
        nodes[key] = (rs[keep], Ws[keep], cs[keep])
        max_slab = max(max_slab, key[0])
    return FitTable(z_lo, nz_slab, ntheta, max_slab + 1, nodes)


def _eval_bin(node, r, extrap=0.5):
    """Piecewise-linear W(r) for one bin's nodes; returns (W, conf) arrays."""
    rn, Wn, cn = node
    if len(rn) == 1:
        W = Wn[0] + (r - rn[0]) * 0.0
        conf = cn[0] * np.exp(-np.abs(r - rn[0]) / 4.0)
        return W, conf
    W = np.interp(r, rn, Wn)
    # bounded extrapolation past ends
    lo, hi = rn[0], rn[-1]
    sl_lo = (Wn[1] - Wn[0]) / max(rn[1] - rn[0], 1e-3)
    sl_hi = (Wn[-1] - Wn[-2]) / max(rn[-1] - rn[-2], 1e-3)
    sl_lo, sl_hi = (float(np.clip(s, 1e-4, 0.5)) for s in (sl_lo, sl_hi))
    W = np.where(r < lo, Wn[0] + (r - lo) * sl_lo, W)
    W = np.where(r > hi, Wn[-1] + (r - hi) * sl_hi, W)
    past = np.maximum(np.maximum(lo - r, r - hi), 0.0)
    cint = np.interp(r, rn, cn)
    conf = cint * np.exp(-past / 4.0)
    conf = np.where((past * sl_hi) > extrap, 0.0, conf)
    return W, conf


_RG = {}


def _render_init(table_path, umb, store_path, r_core, block):
    from labelstore import LabelStore
    _RG.update(table=FitTable.load(table_path), umb=umb,
               store=LabelStore.open(store_path), r_core=r_core, block=block)


def _render_one(blk):
    return _render_block(_RG["table"], _RG["umb"], _RG["store"], _RG["r_core"],
                         blk)


def render_region(table, umb, store, r_core=None, workers=0, block=128,
                  table_path=None):
    """Render W/conf densely; normal = grad W by central differences on the
    dense field (continuous — no branch-cut repair needed). Blocks are chunk-
    aligned (disjoint -> race-free parallel writes); conf 0 in the umbilicus
    core and where <2 of 4 bins usable."""
    from ctio import iter_blocks
    if r_core is None:
        pitches = []
        for (rn, Wn, cn) in table.nodes.values():
            if len(rn) > 1:
                pitches += list(np.diff(rn) / np.maximum(np.diff(Wn), 1e-3))
        r_core = 0.75 * float(np.median(pitches)) if pitches else 8.0
    blocks = list(iter_blocks(store.shape, block=block, halo=1))
    if workers > 1 and table_path:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        stats = {"blocks": 0, "labeled_frac": 0.0}
        with ctx.Pool(workers, initializer=_render_init,
                      initargs=(table_path, umb, store.path, r_core, block)) as p:
            for frac in p.imap_unordered(_render_one, blocks, chunksize=4):
                stats["blocks"] += 1
                stats["labeled_frac"] += frac
                if stats["blocks"] % 200 == 0:
                    print(f"  render {stats['blocks']}/{len(blocks)}")
        stats["labeled_frac"] /= max(stats["blocks"], 1)
        return stats
    stats = {"blocks": 0, "labeled_frac": 0.0}
    for blk in blocks:
        stats["blocks"] += 1
        stats["labeled_frac"] += _render_block(table, umb, store, r_core, blk)
    stats["labeled_frac"] /= max(stats["blocks"], 1)
    return stats


def _render_block(table, umb, store, r_core, blk):
    sl = blk.halo
    z, y, x = np.meshgrid(*[np.arange(s.start, s.stop, dtype=np.float32)
                            for s in sl], indexing="ij")
    ctr = umbilicus_at(umb, z.ravel()).reshape(z.shape + (2,))
    ry, rx = y - ctr[..., 0], x - ctr[..., 1]
    r = np.hypot(ry, rx)
    th = np.arctan2(ry, rx)
    tpos = (th + np.pi) / (2 * np.pi) * table.ntheta - 0.5
    spos = (z - table.z_lo) / table.nz_slab - 0.5
    Wacc = np.zeros(z.shape, np.float32)
    Cacc = np.zeros(z.shape, np.float32)
    Nuse = np.zeros(z.shape, np.int16)
    for ds in (0, 1):
        for dt in (0, 1):
            sb = np.floor(spos).astype(np.int64) + ds
            tb = (np.floor(tpos).astype(np.int64) + dt) % table.ntheta
            fw = (1 - np.abs(spos - sb)) * (1 - np.abs(tpos - np.floor(tpos) - dt))
            fw = np.clip(fw, 0, 1)
            # evaluate per distinct bin key present in this block
            keys = np.stack([sb, tb], -1).reshape(-1, 2)
            uk, inv = np.unique(keys, axis=0, return_inverse=True)
            Wv = np.zeros(r.size, np.float32)
            Cv = np.zeros(r.size, np.float32)
            rf = r.ravel()
            for ki, key in enumerate(uk):
                node = table.nodes.get((int(key[0]), int(key[1])))
                if node is None:
                    continue
                m = inv == ki
                Wv[m], Cv[m] = _eval_bin(node, rf[m])
            Wv = Wv.reshape(r.shape)
            Cv = Cv.reshape(r.shape) * fw
            Wacc += Wv * Cv
            Cacc += Cv
            Nuse += (Cv > 1e-6).astype(np.int16)
    ok = (Cacc > 1e-6) & (Nuse >= 2) & (r > r_core)
    Wf = np.where(ok, Wacc / np.maximum(Cacc, 1e-9), 0.0)
    conf = np.where(ok, np.clip(Cacc / 2.0, 0, 1), 0.0)
    gz, gy, gx = np.gradient(Wf.astype(np.float64))
    nrm = np.sqrt(gz * gz + gy * gy + gx * gx)
    degen = (nrm < 1e-4) | (nrm > 0.5) | ~ok
    conf = np.where(degen, 0.0, conf).astype(np.float32)
    nrm = np.maximum(nrm, 1e-9)
    core = blk.core
    cut = tuple(slice(core[i].start - sl[i].start,
                      core[i].stop - sl[i].start) for i in range(3))
    org = tuple(c.start for c in core)
    ang = 2 * np.pi * Wf[cut]
    store.write_block("w_sin", org, np.sin(ang).astype(np.float32))
    store.write_block("w_cos", org, np.cos(ang).astype(np.float32))
    store.write_block("w_conf", org, conf[cut])
    for name, g in (("normal_z", gz), ("normal_y", gy), ("normal_x", gx)):
        store.write_block(name, org, (g / nrm)[cut].astype(np.float32))
    return float((conf[cut] > 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", required=True)
    ap.add_argument("--umbilicus", required=True)
    ap.add_argument("--bbox", type=int, nargs=6, required=True,
                    metavar=("z0", "y0", "x0", "D", "H", "W"))
    ap.add_argument("--scroll", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--voxel-um", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ntheta", type=int, default=720)
    ap.add_argument("--nz-slab", type=float, default=4.0)
    ap.add_argument("--report", default="")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="uniform scale from constraint grid to target grid "
                         "(applied to positions AND umbilicus; theta invariant)")
    args = ap.parse_args()
    files = sorted(glob.glob(args.constraints))
    cons = _load_constraints(files)
    print(f"{len(cons)} constraint sets, "
          f"{sum(len(c['pos']) for c in cons)/1e6:.2f}M points")
    umb = load_umbilicus(args.umbilicus) * args.scale
    z0, y0, x0, D, H, Wd = args.bbox
    # scale, then shift constraints into region-local coords
    for c in cons:
        c["pos"] = c["pos"] * args.scale - np.asarray([z0, y0, x0], np.float32)
    umb_local = umb.copy()
    umb_local[:, 0] -= z0
    umb_local[:, 1] -= y0
    umb_local[:, 2] -= x0
    bins = _bin_points(cons, umb_local, 0.0, args.nz_slab, args.ntheta)
    offsets, residuals, conflicts = solve_mesh_offsets(cons, bins)
    placed = sum(k is not None for k in offsets)
    print(f"offsets: {placed}/{len(cons)} meshes placed; "
          f"{len(residuals)} loop edges, {len(conflicts)} conflicts")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    table = build_table(cons, bins, offsets, 0.0, args.nz_slab, args.ntheta,
                        conflicts)
    table_path = os.path.join(os.path.dirname(args.out) or ".",
                              "fit_" + os.path.basename(args.out) + ".npz")
    table.save(table_path)
    store = LabelStore.create(
        args.out, scroll_id=args.scroll, level=args.level, voxel_um=args.voxel_um,
        origin_zyx=(z0, y0, x0), shape_zyx=(D, H, Wd),
        channels=["w_sin", "w_cos", "w_conf", "normal_z", "normal_y", "normal_x"],
        provenance={"producer": "fitfield.py", "constraints": args.constraints,
                    "n_meshes": len(cons)}, overwrite=args.overwrite)
    stats = render_region(table, umb_local, store, workers=args.workers,
                          table_path=table_path)
    store.finalize()
    print(f"rendered: {stats}")
    if args.report:
        with open(args.report, "w") as f:
            json.dump({"offsets": {cons[i]['id']: offsets[i]
                                   for i in range(len(cons))},
                       "loop_residuals": residuals, "conflicts": conflicts,
                       "render": stats}, f, indent=1)


if __name__ == "__main__":
    main()
