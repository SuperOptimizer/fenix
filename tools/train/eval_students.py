#!/usr/bin/env python3
"""Standard held-out eval for surface students. Generalizes m8_eval.py: N models, and the
metric set carries the doctrine lessons —

  - dice + surface-dice@2 per crop (as M8), on labeled voxels only
  - MEAN and the TAIL (min, p10): the codec sweep proved the mean hides cliffs; damage
    concentrates in thin/faint-sheet crops
  - ECE (expected calibration error, 15 bins) on labeled voxels: the tracer THRESHOLDS
    the probability field, so a well-ranked but miscalibrated model traces badly

Usage:
  eval_students.py [--crops DIR] [--out results.json] NAME=ckpt.pt[:base] ...
  e.g. eval_students.py M=/tmp/gtqc/real/studentM_final.pt:32 G=/tmp/gtqc/m8/studentG_final.pt:16
"""
import sys, os, glob, json, argparse
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import StudentUNet

DEV = "cuda:0"
PATCH, STRIDE = 128, 64


def load_student(ckpt, base):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st.get("ema", st.get("net")) if isinstance(st, dict) else st
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    net = StudentUNet(base=base)
    net.load_state_dict(sd)
    return net.to(DEV).eval()


def gauss3(n, sigma_frac=0.25):
    ax = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * (n * sigma_frac) ** 2)).astype(np.float32)
    return torch.from_numpy(g[:, None, None] * g[None, :, None] * g[None, None, :])


@torch.no_grad()
def predict(net, ct):
    D, H, W = ct.shape
    x = torch.from_numpy(ct.astype(np.float32) / 255.0).to(DEV)
    prob = torch.zeros((D, H, W), device=DEV)
    wsum = torch.zeros((D, H, W), device=DEV)
    w = gauss3(PATCH).to(DEV)
    zs = sorted(set(list(range(0, max(D - PATCH, 0) + 1, STRIDE)) + [max(D - PATCH, 0)]))
    for z0 in zs:
        for y0 in zs:
            for x0 in zs:
                p = x[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH][None, None]
                with torch.autocast("cuda", torch.bfloat16):
                    lo = net(p)
                pr = torch.softmax(lo.float(), 1)[0, 1]
                prob[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] += pr * w
                wsum[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] += w
    return (prob / wsum.clamp_min(1e-6)).cpu().numpy()


def surf_pts(mask):
    return mask & ~ndimage.binary_erosion(mask)


def surface_dice(a, b, tau=2.0):
    if not a.any() or not b.any():
        return 0.0 if (a.any() != b.any()) else 1.0
    sa, sb = surf_pts(a), surf_pts(b)
    da = ndimage.distance_transform_edt(~sa)
    db = ndimage.distance_transform_edt(~sb)
    return float((db[sa] <= tau).sum() + (da[sb] <= tau).sum()) / float(sa.sum() + sb.sum())


def ece(prob, sheet, lab, bins=15):
    """Expected calibration error over labeled voxels (sheet = positive class)."""
    p = prob[lab]
    y = sheet[lab].astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e, n = 0.0, len(p)
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        e += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return float(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="/tmp/gtqc/m8/eval")
    ap.add_argument("--out", default="/tmp/gtqc/eval_results.json")
    ap.add_argument("models", nargs="+", help="NAME=ckpt.pt[:base] (base default 16)")
    args = ap.parse_args()

    nets = {}
    for spec in args.models:
        name, rest = spec.split("=", 1)
        ckpt, _, base = rest.partition(":")
        nets[name] = load_student(ckpt, int(base) if base else 16)

    rows = []
    for cd in sorted(glob.glob(f"{args.crops}/crop*")):
        ct = np.load(f"{cd}/ct.npy")
        gt = np.load(f"{cd}/gt.npy")
        lab = gt > 0
        sheet = gt == 255
        if sheet.sum() < 1000:
            print(f"{cd}: skipping (only {sheet.sum()} sheet voxels)")
            continue
        r = {"crop": os.path.basename(cd), "n_sheet": int(sheet.sum()), "n_lab": int(lab.sum())}
        for name, net in nets.items():
            prob = predict(net, ct)
            pred = (prob > 0.5) & lab
            inter = (pred & sheet).sum()
            r[f"dice_{name}"] = float(2 * inter / max(pred.sum() + sheet.sum(), 1))
            r[f"sd2_{name}"] = surface_dice(pred, sheet & lab, 2.0)
            r[f"ece_{name}"] = ece(prob, sheet, lab)
        rows.append(r)
        print(f"{r['crop']}: " + " | ".join(
            f"{n} dice {r[f'dice_{n}']:.3f} sd2 {r[f'sd2_{n}']:.3f} ece {r[f'ece_{n}']:.3f}"
            for n in nets), flush=True)

    summary = {}
    for name in nets:
        for m in ("dice", "sd2", "ece"):
            v = np.array([r[f"{m}_{name}"] for r in rows])
            summary[f"{m}_{name}"] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                                      "min": float(v.min())}
            print(f"{name} {m}: mean {v.mean():.4f}  p10 {np.percentile(v,10):.4f}  min {v.min():.4f}")
    json.dump({"crops": rows, "summary": summary}, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
