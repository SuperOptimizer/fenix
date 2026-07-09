"""Per-crop GT quality grading — sidesteps the whole-scroll WAN wall.

Published segments span nearly the whole scroll (z 2k-72k), so QC'ing them against remote
CT thrashes. Instead: sample N uv-region CROPS of a segment; each crop's CT bounding box is
tiny (a 32x32 uv crop ~= 670^3 vox) and LOCAL. Ingest just that little box, QC the crop at
full res, aggregate. Per-crop grades also localize WHERE registration is good/bad (the D
trust-mask case), which whole-mesh scalars hide.

Reads the tifxyz directly (x/y/z.tif, LOD-0 coords via the -on-<scan> mesh). For each crop:
ingest CT box -> import crop as a mini .fxsurf -> surf-qc (profile) -> grade.

Plan: docs/design/gt-autograde-improve.md. Companion to grade_corpus.py (whole-mesh) and
the downscale path.

Usage:
  crop_qc.py --tifxyz <url-or-dir> --zarr <url> --crops 12 --uv 48 [--out card.json]
"""
import sys, os, json, argparse, subprocess, tempfile, re
import numpy as np, tifffile

FENIX = "/home/forrest/fenix/build-release/fenix"


def run(cmd, timeout=300):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout + r.stderr


def fetch_tif(base, name, tmp):
    out = f"{tmp}/{name}"
    if base.startswith("http") or base.startswith("s3"):
        run(["curl", "-s", "--max-time", "180", f"{base.rstrip('/')}/{name}", "-o", out])
    else:
        out = f"{base.rstrip('/')}/{name}"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tifxyz", required=True, help="-on-<scan> tifxyz dir/url (LOD-0 coords)")
    ap.add_argument("--zarr", required=True, help="matching CT zarr MULTISCALE ROOT url (not /0)")
    ap.add_argument("--crops", type=int, default=12)
    ap.add_argument("--uv", type=int, default=48, help="uv crop side (cells)")
    ap.add_argument("--pad", type=int, default=24, help="CT box padding (vox)")
    ap.add_argument("--out", default="/tmp/gtqc/cropcard.json")
    ap.add_argument("--tmp", default="/tmp/gtqc/crops")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)
    zroot = re.sub(r"/0/?$", "", args.zarr)  # ingest-zarr wants the multiscale root + level arg

    with tempfile.TemporaryDirectory(dir=args.tmp) as tmp:
        xp = fetch_tif(args.tifxyz, "x.tif", tmp)
        yp = fetch_tif(args.tifxyz, "y.tif", tmp)
        zp = fetch_tif(args.tifxyz, "z.tif", tmp)
        X = tifffile.imread(xp).astype(np.float32)
        Y = tifffile.imread(yp).astype(np.float32)
        Z = tifffile.imread(zp).astype(np.float32)
        nv, nu = X.shape
        valid = (X > 0) & (Y > 0) & (Z > 0)
        print(f"mesh uv {X.shape}, {valid.sum()} valid; sampling {args.crops} {args.uv}x{args.uv} crops")

        rng = np.random.default_rng(0)
        # pick crop origins whose window has enough valid cells
        cands = []
        for _ in range(args.crops * 6):
            u0 = int(rng.integers(0, max(1, nu - args.uv)))
            v0 = int(rng.integers(0, max(1, nv - args.uv)))
            m = valid[v0:v0+args.uv, u0:u0+args.uv]
            if m.sum() > args.uv * args.uv * 0.3:
                cands.append((u0, v0, int(m.sum())))
            if len(cands) >= args.crops:
                break

        crops = []
        for i, (u0, v0, ncell) in enumerate(cands):
            zc = Z[v0:v0+args.uv, u0:u0+args.uv]; yc = Y[v0:v0+args.uv, u0:u0+args.uv]; xc = X[v0:v0+args.uv, u0:u0+args.uv]
            m = (zc > 0) & (yc > 0) & (xc > 0)
            z0, z1 = int(zc[m].min())-args.pad, int(zc[m].max())+args.pad
            y0, y1 = int(yc[m].min())-args.pad, int(yc[m].max())+args.pad
            x0, x1 = int(xc[m].min())-args.pad, int(xc[m].max())+args.pad
            z0, y0, x0 = max(0, z0), max(0, y0), max(0, x0)
            box = (z1-z0, y1-y0, x1-x0)
            fx = f"{tmp}/c{i}.fxvol"
            r = run([FENIX, "ingest-zarr", zroot, "0", str(z0), str(y0), str(x0),
                     str(box[0]), str(box[1]), str(box[2]), fx, "q=1"])
            if "wrote" not in r:
                print(f"  crop {i}: ingest failed"); continue
            # build a mini tifxyz for just this crop, coords SHIFTED into the box frame
            cdir = f"{tmp}/c{i}tif"; os.makedirs(cdir, exist_ok=True)
            tifffile.imwrite(f"{cdir}/x.tif", np.where(m, xc - x0, -1).astype(np.float32))
            tifffile.imwrite(f"{cdir}/y.tif", np.where(m, yc - y0, -1).astype(np.float32))
            tifffile.imwrite(f"{cdir}/z.tif", np.where(m, zc - z0, -1).astype(np.float32))
            open(f"{cdir}/meta.json", "w").write('{"scale":[1,1],"format":"tifxyz","type":"seg"}')
            cfx = f"{tmp}/c{i}.fxsurf"
            run([FENIX, "import-tifxyz", cdir, cfx])
            qc = run([FENIX, "surf-qc", fx, cfx, "k=80", "off=12", "profile=1"])
            prof = {}
            for key, pat in [("n_offs", r"n_offs=(\d+)"), ("pap", r"on-papyrus (\d+)%"), ("ridge", r"ridge (\d+)%"), ("med_off", r"median-offset (-?\d+)"),
                             ("iqr", r"offset-IQR (-?\d+)"), ("coherent", r"coherent (\d+)%"),
                             ("air", r"AIR (\d+)%")]:
                g = re.search(pat, qc)
                if g:
                    prof[key] = int(g.group(1))
            crops.append({"u0": u0, "v0": v0, "ncell": ncell, "box": box, **prof})
            print(f"  crop {i} uv({u0},{v0}) box{box}: ridge {prof.get('ridge','?')}% "
                  f"iqr {prof.get('iqr','?')} coh {prof.get('coherent','?')}%")

        # aggregate: per-segment grade from crop distribution
        rid = [c.get("ridge", 0) for c in crops]
        # iqr FAILS CLOSED: -1 = unmeasured (too few peak-firing points); exclude those and
        # low-n_offs crops from the dispersion aggregate rather than treating them as tight.
        iqr = [c["iqr"] for c in crops if c.get("iqr", -1) >= 0 and c.get("n_offs", 99) >= 20]
        paps = [c["pap"] for c in crops if "pap" in c]
        cohs = [c["coherent"] for c in crops if c.get("coherent", -1) >= 0]
        meds = [c["med_off"] for c in crops if "med_off" in c]
        card = {"tifxyz": args.tifxyz, "n_crops": len(crops),
                "pap_med": float(np.median(paps)) if paps else None,
                "coh_med": float(np.median(cohs)) if cohs else None,
                "med_off_med": float(np.median(meds)) if meds else None,
                "n_iqr_crops": len(iqr),
                "ridge_med": float(np.median(rid)) if rid else 0,
                "iqr_med": float(np.median(iqr)) if iqr else None,
                "frac_crops_good": float(np.mean([r >= 70 for r in rid])) if rid else 0,
                "crops": crops}
        json.dump(card, open(args.out, "w"), indent=2)
        print(f"\nSEGMENT: ridge_med {card['ridge_med']:.0f}% iqr_med {card['iqr_med']:.0f} "
              f"good-crop frac {card['frac_crops_good']:.2f} -> {args.out}")


if __name__ == "__main__":
    main()
