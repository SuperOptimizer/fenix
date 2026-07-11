"""M5: recto/verso orientation metric — a failure mode nothing else measures.

Per mesh: unit normals from the uv grid (central differences), dotted with the radial-
outward direction from the scroll axis (proxy: scan-center y/x, z-free). Reports:
  consistency = max fraction of cells sharing a sign  (mixed signs = damaged normal field)
  majority    = +1 outward / -1 inward               (convention flip vs corpus majority)

A healthy recto trace has consistency ~1.0 and the corpus-majority sign. compose() caps
mixed-orientation meshes at C (plan M5, gt-metrics-hardening.md).

Usage: orientation_check.py --tifxyz <dir-or-url> [--center y,x] [--out o.json]
Self-test: --selftest flips the mesh's u axis (normals invert) and verifies detection.
"""
import sys, os, json, argparse, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, tifffile
from fault_inject import load_tifxyz, normals


def load_axis(toml_path):
    """Parse the umbilicus TOML (z/y/x control arrays) -> per-z interpolator."""
    import re as _re
    txt = open(toml_path).read()
    def arr(name):
        m = _re.search(rf"^{name} = \[(.*?)\]", txt, _re.M | _re.S)
        return np.array([float(t) for t in m.group(1).split(",")]) if m else None
    z, y, x = arr("z"), arr("y"), arr("x")
    return lambda zq: (np.interp(zq, z, y), np.interp(zq, z, x))


def orientation(X, Y, Z, V, nm, center_fn, k=4000):
    vv, uu = np.where(V)
    if len(vv) == 0:
        return None
    rng = np.random.default_rng(0)
    sel = rng.choice(len(vv), min(k, len(vv)), replace=False)
    vv, uu = vv[sel], uu[sel]
    n = nm[vv, uu]                               # [k,3] ZYX
    ok = np.linalg.norm(n, axis=1) > 0.5
    vv, uu, n = vv[ok], uu[ok], n[ok]
    cy, cx = center_fn(Z[vv, uu])                # curved-axis center at each point's z
    rad = np.stack([np.zeros(len(vv)), Y[vv, uu] - cy, X[vv, uu] - cx], 1)
    rn = np.linalg.norm(rad, axis=1); rn[rn < 1e-6] = 1
    d = (n * (rad / rn[:, None])).sum(1)
    sgn = np.sign(d[np.abs(d) > 0.2])            # ignore grazing (normal ~tangential to radius)
    if len(sgn) < 50:
        return {"consistency": None, "majority": 0, "n": int(len(sgn))}
    fpos = float((sgn > 0).mean())
    return {"consistency": float(max(fpos, 1 - fpos)), "majority": 1 if fpos >= 0.5 else -1,
            "n": int(len(sgn))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifxyz", required=True)
    ap.add_argument("--axis", default=None, help="umbilicus TOML (fenix umbilicus output) — the real curved axis")
    ap.add_argument("--center", default=None, help="fallback fixed y,x axis proxy")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jsonl", default=None,
                    help="append {segment, consistency, majority} to this jsonl (well-formed "
                         "JSON from here, not shell regex — a '+1' majority written by shell "
                         "was invalid JSON and crashed scorecard)")
    ap.add_argument("--seg", default=None, help="segment name for --jsonl (default: tifxyz dir basename)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        X, Y, Z, V = load_tifxyz(args.tifxyz, tmp)
    if args.axis:
        center_fn = load_axis(args.axis)
    elif args.center:
        cy0, cx0 = [float(t) for t in args.center.split(",")]
        center_fn = lambda zq: (np.full_like(np.asarray(zq, float), cy0), np.full_like(np.asarray(zq, float), cx0))
    else:
        cy0, cx0 = float(np.median(Y[V])), float(np.median(X[V]))
        center_fn = lambda zq: (np.full_like(np.asarray(zq, float), cy0), np.full_like(np.asarray(zq, float), cx0))
    nm = normals(X, Y, Z, V)
    o = orientation(X, Y, Z, V, nm, center_fn)
    print(f"orientation: consistency {o['consistency']}  majority {o['majority']:+d}  (n={o['n']})")
    if args.jsonl and o["consistency"] is not None:
        seg = args.seg or os.path.basename(os.path.normpath(args.tifxyz))
        with open(args.jsonl, "a") as jf:
            jf.write(json.dumps({"segment": seg, "consistency": o["consistency"],
                                 "majority": int(o["majority"])}) + "\n")
    if args.selftest:
        Xf, Yf, Zf, Vf = X[:, ::-1].copy(), Y[:, ::-1].copy(), Z[:, ::-1].copy(), V[:, ::-1].copy()
        of = orientation(Xf, Yf, Zf, Vf, normals(Xf, Yf, Zf, Vf), center_fn)
        flipped = of["majority"] == -o["majority"] and (of["consistency"] or 0) > 0.8
        print(f"selftest (u-axis flipped): majority {of['majority']:+d} consistency {of['consistency']}"
              f"  -> {'PASS (flip detected)' if flipped else 'FAIL'}")
    if args.out:
        json.dump(o, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
