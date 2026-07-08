"""solve — out-of-core whole-scroll unwrap / stitch / emit (the tracer at scale).

Scales unwrap.py (single source of truth for the PCG unwrap) to a whole volume:

  pass 1 unwrap: per block (core B + halo), weighted-LS unwrap, phase-align so
    U mod 1 ~= w, store rel = round(U - w) as int16 kfield (2B/voxel, and
    piecewise-constant so it compresses 20-50x; f32 U is never materialized)
    + face slabs for stitching. Multiprocessing, resume via face-file markers.
  pass 2 stitch: median U-difference per adjacent face -> integer edges;
    union-find components; MAX-WEIGHT spanning tree (one bad face can't poison
    a BFS tree); loop residuals; weighted LS over the cycle space if any
    residual != 0; conflict regions reported for fit arbitration.
  pass 3 emit: wrap = rel + off + C + (w >= 0.5), umbilicus-anchored C
    (innermost wrap ~ 0), int16 wrap_id zarr + per-wrap summary.

  python solve.py all --preds P.zarr --out S/ [--umbilicus u.txt --anno-scale 4]
  python solve.py eval-phantom [--size 256]     # e2e gate on synthetic oracle
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from unwrap import unwrap_block, wrapdiff

SENT = -32768


# ---- grid ----------------------------------------------------------------
def block_keys(shape, B):
    for bz in range((shape[0] + B - 1) // B):
        for by in range((shape[1] + B - 1) // B):
            for bx in range((shape[2] + B - 1) // B):
                yield (bz, by, bx)


def core_slices(shape, B, key):
    return tuple(slice(k * B, min((k + 1) * B, shape[i]))
                 for i, k in enumerate(key))


def window_slices(shape, B, halo, key):
    c = core_slices(shape, B, key)
    return tuple(slice(max(c[i].start - halo, 0), min(c[i].stop + halo, shape[i]))
                 for i in range(3))


# ---- IO --------------------------------------------------------------------
class PredReader:
    def __init__(self, path):
        import zarr
        self.g = zarr.open_group(path, mode="r", zarr_format=2)
        with open(os.path.join(path, "meta.json")) as f:
            self.meta = json.load(f)
        self.shape = tuple(self.meta["shape_zyx"])
        self.conf_ch = "interior" if "interior" in self.meta["channels"] \
            else "w_conf"

    def read_wc(self, sl):
        s = (np.asarray(self.g["w_sin"][sl], np.float32) - 127.5) / 127.5
        c = (np.asarray(self.g["w_cos"][sl], np.float32) - 127.5) / 127.5
        conf = np.asarray(self.g[self.conf_ch][sl], np.float32) / 255.0
        w = (np.arctan2(s, c) / (2 * np.pi)) % 1.0
        return w.astype(np.float32), conf


def _i16_store(path, shape, chunks=(128,) * 3, mode="a"):
    import zarr
    from numcodecs import Blosc
    g = zarr.open_group(os.path.dirname(path) or ".", mode="a", zarr_format=2)
    name = os.path.basename(path)
    if name not in g:
        g.create_array(name, shape=shape, chunks=chunks, dtype="i2",
                       fill_value=SENT, chunk_key_encoding={"name": "v2", "separator": "/"},
                       compressors=Blosc(cname="zstd", clevel=5,
                                         shuffle=Blosc.BITSHUFFLE))
    return g[name]


# ---- pass 1: unwrap ----------------------------------------------------------
_G = {}


def _uw_init(cfg):
    _G["cfg"] = cfg
    _G["preds"] = PredReader(cfg["preds"])
    _G["kf"] = _i16_store(os.path.join(cfg["out"], "kfield"),
                          _G["preds"].shape)


def _face_path(out, key):
    return os.path.join(out, "faces", "%d_%d_%d.npz" % key)


def unwrap_one(key):
    cfg = _G["cfg"]
    preds, kf = _G["preds"], _G["kf"]
    fp = _face_path(cfg["out"], key)
    if os.path.exists(fp):
        return {"key": key, "status": "skipped"}
    shape, B, halo = preds.shape, cfg["B"], cfg["halo"]
    win = window_slices(shape, B, halo, key)
    w, conf = preds.read_wc(win)
    m = conf > cfg["conf_thresh"]
    payload = {}
    if m.sum() < cfg["min_vox"]:
        status = "empty"
    else:
        U, resid = unwrap_block(w.astype(np.float64), conf,
                                iters=cfg["iters"], return_resid=True)
        c_align = float(np.median(wrapdiff((w - U) % 1.0)[m]))
        U = U + c_align
        rel = np.round(U - w)
        if np.abs(rel).max() >= 30000:
            status = "rel_overflow"
        else:
            rel = rel.astype(np.int16)
            rel[conf <= cfg["conf_emit"]] = SENT
            core = core_slices(shape, B, key)
            cut = tuple(slice(core[i].start - win[i].start,
                              core[i].stop - win[i].start) for i in range(3))
            kf[core] = rel[cut]
            status = "ok"
            payload["converged"] = resid < 1e-4
            for ax in range(3):
                for side, sl_ax in (("lo", slice(0, 2 * halo)),
                                    ("hi", slice(win[ax].stop - win[ax].start
                                                 - 2 * halo, None))):
                    if (side == "lo" and win[ax].start == 0) or \
                       (side == "hi" and win[ax].stop == shape[ax]):
                        continue
                    sl = [slice(None)] * 3
                    sl[ax] = sl_ax
                    payload[f"U_{side}{ax}"] = U[tuple(sl)].astype(np.float32)
                    payload[f"c_{side}{ax}"] = (conf[tuple(sl)] * 255).astype(np.uint8)
    tmp = fp[:-4] + f".tmp.{os.getpid()}.npz"   # savez appends .npz otherwise
    np.savez_compressed(tmp, status=status,
                        occupancy=float(m.mean()), **payload)
    os.replace(tmp, fp)
    return {"key": key, "status": status}


def run_unwrap(cfg):
    os.makedirs(os.path.join(cfg["out"], "faces"), exist_ok=True)
    preds = PredReader(cfg["preds"])
    keys = list(block_keys(preds.shape, cfg["B"]))
    ctx = mp.get_context("spawn")
    stats = defaultdict(int)
    t0 = time.time()
    with ctx.Pool(cfg["workers"], initializer=_uw_init, initargs=(cfg,),
                  maxtasksperchild=16) as pool:
        for i, r in enumerate(pool.imap_unordered(unwrap_one, keys, chunksize=1)):
            stats[r["status"]] += 1
            if (i + 1) % 20 == 0:
                print(f"  unwrap {i+1}/{len(keys)} {dict(stats)} "
                      f"({(time.time()-t0)/(i+1):.1f}s/blk)")
    return dict(stats)


# ---- pass 2: stitch -----------------------------------------------------------
def _load_face(out, key):
    fp = _face_path(out, key)
    return np.load(fp, allow_pickle=False) if os.path.exists(fp) else None


def run_stitch(cfg):
    preds = PredReader(cfg["preds"])
    B = cfg["B"]
    nb = tuple((preds.shape[i] + B - 1) // B for i in range(3))
    ok = {}
    for key in block_keys(preds.shape, B):
        f = _load_face(cfg["out"], key)
        if f is not None and str(f["status"]) == "ok":
            ok[key] = f
    edges = []
    for a in ok:
        for ax in range(3):
            b = list(a)
            b[ax] += 1
            b = tuple(b)
            if b not in ok:
                continue
            fa, fb = ok[a], ok[b]
            ka, kb = f"U_hi{ax}", f"U_lo{ax}"
            if ka not in fa or kb not in fb:
                continue
            Ua, Ub = fa[ka], fb[kb]
            ca = fa[f"c_hi{ax}"].astype(np.float32) / 255
            cb = fb[f"c_lo{ax}"].astype(np.float32) / 255
            if Ua.shape != Ub.shape:
                continue
            m = (ca > cfg["conf_thresh"]) & (cb > cfg["conf_thresh"])
            if m.sum() < cfg["min_overlap"]:
                continue
            d = float(np.median((Ua - Ub)[m]))
            q = 1.0 - 2.0 * abs(d - round(d))
            w_e = q * min(int(m.sum()), 10 * cfg["min_overlap"])
            for f_, kk in ((fa, a), (fb, b)):
                if not bool(f_["converged"]):
                    w_e *= 0.25
            edges.append({"a": a, "b": b, "ax": ax, "d": d,
                          "d_int": int(round(d)), "q": q, "w": w_e})
    kept = [e for e in edges if e["q"] >= cfg["q_min"]]
    quarantined = [e for e in edges if e["q"] < cfg["q_min"]]
    # union-find
    parent = {k: k for k in ok}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for e in kept:
        parent[find(e["a"])] = find(e["b"])
    comps = defaultdict(list)
    for k in ok:
        comps[find(k)].append(k)
    # max-weight spanning tree per component (Kruskal), then BFS solve
    offsets, comp_of = {}, {}
    for ci, (root, members) in enumerate(comps.items()):
        eset = sorted((e for e in kept if find(e["a"]) == find(root)),
                      key=lambda e: -e["w"])
        par2 = {k: k for k in members}

        def find2(x):
            while par2[x] != x:
                par2[x] = par2[par2[x]]
                x = par2[x]
            return x
        adj = defaultdict(list)
        for e in eset:
            if find2(e["a"]) != find2(e["b"]):
                par2[find2(e["a"])] = find2(e["b"])
                adj[e["a"]].append((e["b"], e["d_int"]))
                adj[e["b"]].append((e["a"], -e["d_int"]))
        x0 = min(members)
        off = {x0: 0}
        stack = [x0]
        while stack:
            u = stack.pop()
            for v, d in adj[u]:
                if v not in off:
                    off[v] = off[u] + d
                    stack.append(v)
        # loop residuals; LS refine if any nonzero
        cedges = [e for e in kept if e["a"] in off and e["b"] in off]
        resid = [e["d_int"] - (off[e["b"]] - off[e["a"]]) for e in cedges]
        if any(resid):
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import lsqr
            idx = {k: i for i, k in enumerate(off)}
            rows, cols, vals, rhs, wts = [], [], [], [], []
            for i, e in enumerate(cedges):
                rows += [i, i]
                cols += [idx[e["b"]], idx[e["a"]]]
                vals += [np.sqrt(e["w"]), -np.sqrt(e["w"])]
                rhs.append(e["d"] * np.sqrt(e["w"]))
            A = coo_matrix((vals, (rows, cols)), shape=(len(cedges), len(idx)))
            sol = lsqr(A, np.asarray(rhs), atol=1e-10, btol=1e-10)[0]
            sol -= sol[idx[x0]]
            off = {k: int(round(sol[idx[k]])) for k in off}
        for k, v in off.items():
            offsets[k] = v
            comp_of[k] = ci
    conflicts = []
    for e in kept:
        if e["a"] in offsets and e["b"] in offsets:
            r = e["d"] - (offsets[e["b"]] - offsets[e["a"]])
            if abs(r - round(r)) > cfg["conflict_tol"] or round(r) != 0:
                conflicts.append({**{k: (list(v) if isinstance(v, tuple) else v)
                                     for k, v in e.items()}, "resid": r})
    for e in quarantined:
        conflicts.append({**{k: (list(v) if isinstance(v, tuple) else v)
                             for k, v in e.items()}, "resid": None,
                          "quarantined": True})
    out = {"components": len(comps),
           "offsets": {"%d_%d_%d" % k: v for k, v in offsets.items()},
           "comp_of": {"%d_%d_%d" % k: v for k, v in comp_of.items()},
           "n_edges": len(edges), "n_kept": len(kept),
           "n_conflicts": len(conflicts)}
    with open(os.path.join(cfg["out"], "offsets.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(cfg["out"], "conflicts.json"), "w") as f:
        json.dump(conflicts, f, indent=1)
    print(f"stitch: {len(ok)} blocks, {len(kept)}/{len(edges)} edges kept, "
          f"{len(comps)} components, {len(conflicts)} conflicts")
    return out


# ---- pass 3: emit --------------------------------------------------------------
def run_emit(cfg):
    preds = PredReader(cfg["preds"])
    kf = _i16_store(os.path.join(cfg["out"], "kfield"), preds.shape)
    wid = _i16_store(os.path.join(cfg["out"], "wrap_id"), preds.shape)
    with open(os.path.join(cfg["out"], "offsets.json")) as f:
        st = json.load(f)
    offsets = {tuple(int(v) for v in k.split("_")): o
               for k, o in st["offsets"].items()}
    B = cfg["B"]
    # anchor: umbilicus annulus median -> C
    C = 0
    if cfg.get("umbilicus"):
        from mesh_ingest import load_umbilicus, umbilicus_at
        poly = load_umbilicus(cfg["umbilicus"]) / cfg.get("anno_scale", 1.0)
        samples = []
        for key, off in offsets.items():
            core = core_slices(preds.shape, B, key)
            zc = np.arange(core[0].start, core[0].stop, 8)
            cyx = umbilicus_at(poly, zc)
            inside = ((cyx[:, 0] >= core[1].start) & (cyx[:, 0] < core[1].stop)
                      & (cyx[:, 1] >= core[2].start) & (cyx[:, 1] < core[2].stop))
            if not inside.any():
                continue
            w, conf = preds.read_wc(core)
            rel = np.asarray(kf[core])
            for zi in np.nonzero(inside)[0]:
                z = int(zc[zi] - core[0].start)
                y = int(cyx[zi, 0] - core[1].start)
                x = int(cyx[zi, 1] - core[2].start)
                r0, r1 = int(cfg.get("r0", 2)), int(cfg.get("r1", 6))
                sl = (z, slice(max(y - r1, 0), y + r1), slice(max(x - r1, 0), x + r1))
                mm = (conf[sl] > cfg["conf_thresh"]) & (rel[sl] != SENT)
                if mm.any():
                    samples += list((w[sl] + rel[sl] + off)[mm])
            if len(samples) > 1e6:
                break
        if samples:
            C = -int(round(float(np.median(samples))))
            print(f"anchor: C={C} from {len(samples)} umbilicus samples")
        else:
            print("anchor: umbilicus gave no confident samples — min-shift")
    if C == 0 and offsets:
        rels = []
        for key in list(offsets)[:64]:
            core = core_slices(preds.shape, B, key)
            r = np.asarray(kf[core])
            v = r[r != SENT]
            if v.size:
                rels.append(int(v.min()) + offsets[key])
        if rels:
            C = -min(rels)
    counts = defaultdict(int)
    for key, off in offsets.items():
        core = core_slices(preds.shape, B, key)
        rel = np.asarray(kf[core])
        w, conf = preds.read_wc(core)
        wrap = rel.astype(np.int32) + off + C + (w >= 0.5).astype(np.int32)
        wrap[(rel == SENT) | (conf <= cfg["conf_emit"])] = SENT
        wid[core] = np.clip(wrap, SENT, 32767).astype(np.int16)
        u, n = np.unique(wrap[wrap != SENT], return_counts=True)
        for uu, nn in zip(u, n):
            counts[int(uu)] += int(nn)
    with open(os.path.join(cfg["out"], "wraps_summary.json"), "w") as f:
        json.dump({"C": C, "wraps": {str(k): v for k, v in
                                     sorted(counts.items())}}, f, indent=1)
    print(f"emit: {len(counts)} wraps, C={C}")
    return {"C": C, "n_wraps": len(counts)}


# ---- e2e phantom eval ------------------------------------------------------------
def eval_phantom(size=256, B=128, halo=8, noise=0.15, seed=0, workers=4):
    """Full pipeline gate on a synthetic oracle: phantom -> fake 'predictions'
    (noisy sin/cos + interior conf) -> unwrap/stitch/emit -> wrap-id accuracy."""
    import shutil
    import tempfile

    from phantom import make_phantom
    td = tempfile.mkdtemp(prefix="solve_eval_",
                          dir=os.environ.get("SCRATCH", "/tmp"))
    try:
        ph = make_phantom(size, seed=seed)
        rng = np.random.default_rng(seed)
        ang = 2 * np.pi * ph["w"]
        s = np.sin(ang) + rng.normal(0, noise, ang.shape)
        c = np.cos(ang) + rng.normal(0, noise, ang.shape)
        interior = ph["wmask"].astype(np.float32)
        from ctio import PredStore
        preds_path = os.path.join(td, "preds.zarr")
        store = PredStore.create(preds_path, scroll_id="phantom", level=0,
                                 voxel_um=1.0, shape_zyx=(size,) * 3,
                                 channels=["w_sin", "w_cos", "interior"],
                                 provenance={"eval": True})
        store.write_brick((0, 0, 0), {
            "w_sin": np.clip(s, -1, 1).astype(np.float32),
            "w_cos": np.clip(c, -1, 1).astype(np.float32),
            "interior": interior})
        out = os.path.join(td, "solve")
        os.makedirs(out, exist_ok=True)
        cfg = {"preds": preds_path, "out": out, "B": B, "halo": halo,
               "conf_thresh": 0.5, "conf_emit": 0.25, "min_vox": 500,
               "iters": 60, "min_overlap": 200, "q_min": 0.5,
               "conflict_tol": 0.35, "workers": workers}
        print(run_unwrap(cfg))
        run_stitch(cfg)
        run_emit(cfg)
        wid = _i16_store(os.path.join(out, "wrap_id"), (size,) * 3)
        wrap = np.asarray(wid[...])
        pap = ph["papyrus"].astype(bool) & (wrap != SENT)
        gt = np.round(ph["W"]).astype(np.int32)
        # global constant freedom: mode-align
        d = (wrap - gt)[pap]
        shift = int(np.bincount(d - d.min()).argmax()) + d.min()
        acc = float(((wrap - shift) == gt)[pap].mean())
        print(f"EVAL-PHANTOM wrap-id accuracy = {acc:.4f} "
              f"(coverage {pap.mean():.2%}) {'PASS' if acc > 0.99 else 'FAIL'}")
        return acc
    finally:
        shutil.rmtree(td, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("unwrap", "stitch", "emit", "all"):
        p = sub.add_parser(name)
        p.add_argument("--preds", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--block", type=int, default=256)
        p.add_argument("--halo", type=int, default=8)
        p.add_argument("--workers", type=int, default=max(mp.cpu_count() // 2, 1))
        p.add_argument("--conf-thresh", type=float, default=0.5)
        p.add_argument("--conf-emit", type=float, default=0.25)
        p.add_argument("--umbilicus", default="")
        p.add_argument("--anno-scale", type=float, default=1.0)
    e = sub.add_parser("eval-phantom")
    e.add_argument("--size", type=int, default=256)
    e.add_argument("--block", type=int, default=128)
    e.add_argument("--noise", type=float, default=0.15)
    e.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "eval-phantom":
        ok = eval_phantom(args.size, args.block, noise=args.noise,
                          workers=args.workers) > 0.99
        raise SystemExit(0 if ok else 1)
    cfg = {"preds": args.preds, "out": args.out, "B": args.block,
           "halo": args.halo, "conf_thresh": args.conf_thresh,
           "conf_emit": args.conf_emit, "min_vox": 500, "iters": 60,
           "min_overlap": 200, "q_min": 0.5, "conflict_tol": 0.35,
           "workers": args.workers, "umbilicus": args.umbilicus,
           "anno_scale": args.anno_scale}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    if args.cmd in ("unwrap", "all"):
        print(run_unwrap(cfg))
    if args.cmd in ("stitch", "all"):
        run_stitch(cfg)
    if args.cmd in ("emit", "all"):
        run_emit(cfg)


if __name__ == "__main__":
    main()
