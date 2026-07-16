#!/usr/bin/env python3
"""Whole-wrap eval against canonical dense surface labels (scrollprize HF release).

Unlike mesh-raster GT (partial wrap coverage -> precision meaningless), canonical labels
are dense within their region, so BOTH surface-recall and surface-precision are real:
  recall2  = frac of canon sheet voxels within 2 vox of predicted surface (pred > thr)
  prec2    = frac of predicted surface voxels within 2 vox of canon sheet
Scored only where canon is not ignore. Crops ship as <dir>/cropNN/{ct,canon}.npy.

build mode:  canon_eval.py build --out DIR   (samples crops from the label zarrs + HTTP CT)
score mode:  canon_eval.py score --ckpt model.pt[:base] --crops DIR --out results.json
"""
import argparse, glob, json, os, sys, urllib.request
import numpy as np

SETS = [  # (name, label zarr level dir, CT zarr level URL, sheet value, n crops)
    ("0139w23", "/shards/canonical_labels/w023-w032.zarr/0",
     "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/0", 1, 8),
    ("1667", "/shards/canonical_labels/SCROLLS_HEL_2.399um_78keV_0.22m_PHerc_1667_TA_0001_masked_surface.zarr/0",
     "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr/0", 1, 8),
    ("0139w47", "/shards/canonical_labels/w047-w059.zarr/0",
     "https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr/0", 1, 8),
]
S = 192


def fetch_ct(base, o):
    ct = np.zeros((S, S, S), np.uint8)
    for cz in range(o[0] // 128, (o[0] + S + 127) // 128):
        for cy in range(o[1] // 128, (o[1] + S + 127) // 128):
            for cx in range(o[2] // 128, (o[2] + S + 127) // 128):
                try:
                    buf = urllib.request.urlopen(f"{base}/{cz}/{cy}/{cx}", timeout=90).read()
                    a = np.frombuffer(buf, np.uint8).reshape(128, 128, 128)
                except Exception:
                    continue
                z0, y0, x0 = cz * 128 - o[0], cy * 128 - o[1], cx * 128 - o[2]
                zs, ys, xs = max(z0, 0), max(y0, 0), max(x0, 0)
                ze, ye, xe = min(z0 + 128, S), min(y0 + 128, S), min(x0 + 128, S)
                ct[zs:ze, ys:ye, xs:xe] = a[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0]
    return ct


def build(outdir):
    import zarr
    os.makedirs(outdir, exist_ok=True)
    man, ci = [], 0
    for name, lroot, cturl, sheetval, want in SETS:
        z = zarr.open(lroot, mode="r")
        # candidate chunks by compressed size, deterministic spread
        files = sorted((os.path.getsize(p), os.path.basename(p)) for p in glob.glob(f"{lroot}/*")
                       if os.path.basename(p)[0] != "." and os.path.getsize(p) > 60000)
        if not files:
            print(f"{name}: no content chunks, skipping", flush=True)
            continue
        keys = [files[i][1] for i in np.linspace(0, len(files) - 1, want * 3).astype(int)]
        got = 0
        for key in keys:
            if got >= want:
                break
            cz, cy, cx = map(int, key.split("."))
            o = (cz * 128 - 32, cy * 128 - 32, cx * 128 - 32)
            if min(o) < 0:
                continue
            can = np.asarray(z[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S])
            sheet = can == sheetval
            if sheet.mean() < 0.01:
                continue
            ct = fetch_ct(cturl, o)
            if ct.mean() < 20:
                continue
            d = f"{outdir}/crop{ci:02d}"
            os.makedirs(d, exist_ok=True)
            np.save(f"{d}/ct.npy", ct)
            gt = np.where(sheet, np.uint8(1), np.where(can == 2, np.uint8(2), np.uint8(0)))
            np.save(f"{d}/canon.npy", gt)
            man.append({"crop": f"crop{ci:02d}", "set": name, "org": list(o)})
            print(f"crop{ci:02d} {name} org {o} sheet {sheet.mean():.1%}", flush=True)
            ci += 1
            got += 1
    json.dump(man, open(f"{outdir}/manifest.json", "w"), indent=1)
    print(f"BUILD_DONE {ci} crops", flush=True)


def score(ckpt, crops, outpath, thr):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eval_students import load_student, predict
    from scipy.ndimage import binary_dilation
    path, _, base = ckpt.partition(":")
    net = load_student(path, int(base) if base else 16)
    rows = []
    for cd in sorted(glob.glob(f"{crops}/crop*")):
        ct = np.load(f"{cd}/ct.npy")
        gt = np.load(f"{cd}/canon.npy")
        prob = predict(net, ct)
        pred = prob > thr
        sheet, valid = gt == 1, gt != 2
        pd = binary_dilation(pred, iterations=2)
        sd = binary_dilation(sheet, iterations=2)
        r2 = float((sheet & pd).sum() / max(sheet.sum(), 1))
        p2 = float((pred & sd & valid).sum() / max((pred & valid).sum(), 1))
        rows.append({"crop": os.path.basename(cd), "recall2": r2, "prec2": p2,
                     "sheet_frac": float(sheet.mean()), "pred_frac": float(pred.mean())})
        print(f"{os.path.basename(cd)}: recall2 {r2:.3f} prec2 {p2:.3f}", flush=True)
    rs = [r["recall2"] for r in rows]
    ps = [r["prec2"] for r in rows]
    print(f"recall2 mean {np.mean(rs):.3f} std {np.std(rs, ddof=1):.3f} | prec2 mean {np.mean(ps):.3f} | n={len(rows)}")
    json.dump(rows, open(outpath, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "score"])
    ap.add_argument("--out", default="/tmp/gtqc/canon_eval")
    ap.add_argument("--crops", default="/tmp/gtqc/canon_eval")
    ap.add_argument("--ckpt")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    if a.mode == "build":
        build(a.out)
    else:
        score(a.ckpt, a.crops, a.out, a.thr)
