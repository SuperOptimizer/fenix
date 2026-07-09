"""Fault-injection round-trip: the end-to-end validation of the GT grading loop.

Takes a KNOWN-GOOD segment (tifxyz), produces synthetically corrupted copies matching the
failure taxonomy, runs the same crop-QC + grading on original and corruptions, and checks
each lands in its intended bucket — ground truth known by construction, no hand-labels.

  orig        -> should grade A/B
  wrapshift   -> whole mesh moved ~one wrap pitch along normals = WRONG WRAP.
                 MUST NOT grade above orig (the adversarial review's decisive case:
                 pre-P0.3 this graded A on pap~100/iqr~0).
  offset3     -> +3 vox uniform along normals = repairable-uniform (B/C pre-repair)
  warp        -> smooth sinusoidal ±4 vox along normals = repairable-warp (C)
  scramble    -> 20% of points jittered ±6 vox = local damage (D/E territory)

Usage: fault_inject.py --tifxyz <dir-or-url> --zarr <root> [--out /tmp/gtqc/fault]
Plan: docs/design/gt-quality-adversarial-plan.md §3.
"""
import sys, os, json, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tifffile
from scorecard import align_verdict

FENIX = "/home/forrest/fenix/build-release/fenix"


def load_tifxyz(src, tmp):
    def get(name):
        if src.startswith("http"):
            out = f"{tmp}/{name}"
            subprocess.run(["curl", "-s", "--max-time", "180", f"{src.rstrip('/')}/{name}", "-o", out], check=True)
            return out
        return f"{src.rstrip('/')}/{name}"
    X = tifffile.imread(get("x.tif")).astype(np.float32)
    Y = tifffile.imread(get("y.tif")).astype(np.float32)
    Z = tifffile.imread(get("z.tif")).astype(np.float32)
    return X, Y, Z, (X > 0) & (Y > 0) & (Z > 0)


def normals(X, Y, Z, V):
    """Per-cell unit normal from central differences (ZYX cross of uv tangents)."""
    def cd(A, ax):
        g = np.zeros_like(A)
        if ax == 0:
            g[1:-1, :] = (A[2:, :] - A[:-2, :]) * 0.5
        else:
            g[:, 1:-1] = (A[:, 2:] - A[:, :-2]) * 0.5
        return g
    tu = np.stack([cd(Z, 1), cd(Y, 1), cd(X, 1)], -1)   # d/du
    tv = np.stack([cd(Z, 0), cd(Y, 0), cd(X, 0)], -1)   # d/dv
    nm = np.cross(tu, tv)
    nn = np.linalg.norm(nm, axis=-1, keepdims=True)
    nn[nn < 1e-6] = 1
    nm = nm / nn
    nm[~V] = 0
    return nm  # [nv,nu,3] ZYX


def write_tifxyz(d, X, Y, Z, V):
    os.makedirs(d, exist_ok=True)
    tifffile.imwrite(f"{d}/x.tif", np.where(V, X, -1).astype(np.float32))
    tifffile.imwrite(f"{d}/y.tif", np.where(V, Y, -1).astype(np.float32))
    tifffile.imwrite(f"{d}/z.tif", np.where(V, Z, -1).astype(np.float32))
    open(f"{d}/meta.json", "w").write('{"scale":[0.05,0.05],"format":"tifxyz","type":"seg"}')


def corrupt(X, Y, Z, V, nm, kind, rng):
    Xc, Yc, Zc = X.copy(), Y.copy(), Z.copy()
    def shift(mag):  # move along normal by mag (scalar or per-cell array)
        Zc[V] += (nm[..., 0] * mag)[V] if np.ndim(mag) else nm[V][:, 0] * mag
        Yc[V] += (nm[..., 1] * mag)[V] if np.ndim(mag) else nm[V][:, 1] * mag
        Xc[V] += (nm[..., 2] * mag)[V] if np.ndim(mag) else nm[V][:, 2] * mag
    if kind == "wrapshift":
        shift(12.0)                       # ~one wrap pitch at 2.4um
    elif kind == "offset3":
        shift(3.0)
    elif kind == "warp":
        nv, nu = X.shape
        vu, uu = np.meshgrid(np.arange(nv), np.arange(nu), indexing="ij")
        mag = 4.0 * np.sin(2 * np.pi * uu / max(nu, 1) * 3) * np.sin(2 * np.pi * vu / max(nv, 1) * 2)
        shift(mag)
    elif kind == "scramble":
        m = V & (rng.random(X.shape) < 0.20)
        for A in (Zc, Yc, Xc):
            A[m] += rng.uniform(-6, 6, m.sum()).astype(np.float32)
    return Xc, Yc, Zc


def crop_grade(tifdir, zarr, out, crops=8):
    r = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_qc.py"),
                        "--tifxyz", tifdir, "--zarr", zarr, "--crops", str(crops), "--uv", "48",
                        "--out", out], capture_output=True, text=True, timeout=1800)
    if not os.path.exists(out):
        print(r.stdout[-500:], r.stderr[-300:])
        return None, "FAIL"
    card = json.load(open(out))
    tier, _ = align_verdict(card)
    return card, tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifxyz", required=True)
    ap.add_argument("--zarr", required=True)
    ap.add_argument("--out", default="/tmp/gtqc/fault")
    ap.add_argument("--crops", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(7)

    X, Y, Z, V = load_tifxyz(args.tifxyz, args.out)
    nm = normals(X, Y, Z, V)
    print(f"mesh {X.shape}, {V.sum()} valid cells")

    variants = ["orig", "wrapshift", "offset3", "warp", "scramble"]
    results = {}
    for kind in variants:
        d = f"{args.out}/{kind}"
        if kind == "orig":
            write_tifxyz(d, X, Y, Z, V)
        else:
            Xc, Yc, Zc = corrupt(X, Y, Z, V, nm, kind, rng)
            write_tifxyz(d, Xc, Yc, Zc, V)
        card, tier = crop_grade(d, args.zarr, f"{args.out}/{kind}.card.json", args.crops)
        results[kind] = {"tier": tier,
                         "pap": card.get("pap_med") if card else None,
                         "iqr": card.get("iqr_med") if card else None,
                         "coh": card.get("coh_med") if card else None,
                         "med": card.get("med_off_med") if card else None}
        print(f"  {kind:10s} -> tier {tier}  pap {results[kind]['pap']}  iqr {results[kind]['iqr']}  "
              f"coh {results[kind]['coh']}  med {results[kind]['med']}")

    tiers = "ABCDE"
    def rank(t): return tiers.index(t) if t in tiers else 5
    orig, wrap = results["orig"]["tier"], results["wrapshift"]["tier"]
    checks = {
        "orig grades A/B": orig in ("A", "B"),
        "WRAPSHIFT not above orig (decisive)": rank(wrap) >= rank(orig),
        "offset3 repairable tier (B-D)": results["offset3"]["tier"] in ("B", "C", "D"),
        "warp repairable tier (C-D)": results["warp"]["tier"] in ("B", "C", "D"),
        "scramble demoted vs orig": rank(results["scramble"]["tier"]) > rank(orig),
    }
    print("\n=== FAULT-INJECTION VERDICT ===")
    for k, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    json.dump({"results": results, "checks": {k: bool(v) for k, v in checks.items()}},
              open(f"{args.out}/verdict.json", "w"), indent=2)
    print(f"-> {args.out}/verdict.json")


if __name__ == "__main__":
    main()
