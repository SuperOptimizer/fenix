#!/usr/bin/env python3
"""Standard held-out eval for surface students. Generalizes m8_eval.py: N models, and the
metric set carries the doctrine lessons —

  - dice@0.5 + max-dice (threshold-swept over the PR curve, with its argmax threshold)
    + AUPRC + surface-dice@2 per crop, on labeled voxels only. Max-dice/AUPRC are the
    discriminative numbers: the GT-band loss (M10) produces models whose probabilities
    are compressed low — studentM scored dice@0.5 0.001 while tracing fine; its max-dice
    threshold was ~0.01. dice@0.5 stays as the calibration-sensitive operating point.
  - MEAN and the TAIL (min, p10): the codec sweep proved the mean hides cliffs; damage
    concentrates in thin/faint-sheet crops
  - ECE (expected calibration error, 15 bins) on labeled voxels: the tracer THRESHOLDS
    the probability field, so a well-ranked but miscalibrated model traces badly
  - the max-dice threshold is what `trace_eval_run.py --thresh` should be fed (trace-eval
    defaults to 0.10, ~10-30x above a band-loss model's operating point)

Label decode: crop gt.npy round-trips through the DCT codec (gt.fxvol q=0.5), so the
tri-state 255/128/0 comes back SMEARED (253, 129, ringing 1-31 near zero). Exact tests
(gt==255, gt>0) are wrong against it — decode by bands: sheet >=192, background 64..191,
ignore <64 (ringing lands in ignore, where it belongs).

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
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    net = StudentUNet(base=base)
    net.load_state_dict(sd)
    net = net.to(DEV).eval()
    # Fast eval path (measured 2026-07-12, 5060 Ti, 128^3 windows): eager-bf16 18.9
    # ms/window -> +compile 14.2 -> half-weights+channels_last+compile+window-batch-4
    # 11.3 (1.68x total). Pure-fp16 weights are MORE accurate than bf16 autocast here
    # (corr vs fp32: 0.999998 vs 0.999901) — fp16 mantissa beats bf16 at inference.
    # FENIX_EVAL_HALF=0 / FENIX_EVAL_COMPILE=0 opt back out.
    if os.environ.get("FENIX_EVAL_HALF", "1") == "1":
        net = net.half().to(memory_format=torch.channels_last_3d)
        net._fx_half = True
    if os.environ.get("FENIX_EVAL_COMPILE", "1") == "1":
        net = torch.compile(net)
    return net


def gauss3(n, sigma_frac=0.25):
    ax = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * (n * sigma_frac) ** 2)).astype(np.float32)
    return torch.from_numpy(g[:, None, None] * g[None, :, None] * g[None, None, :])


@torch.no_grad()
def predict(net, ct, wbatch=4, scale=1):
    # scale>1: coarse-canon student (canon = 2.4*scale) scored on a 2.4um crop —
    # trilinear downsample CT, predict at the model's grid, upsample probs back.
    # GT stays at 2.4um so recall@2/@4 remain comparable across the model family.
    full_shape = ct.shape
    half = getattr(net, "_fx_half", False) or getattr(getattr(net, "_orig_mod", None), "_fx_half", False)
    x = torch.from_numpy(ct.astype(np.float32) / 255.0).to(DEV)
    if scale > 1:
        x = torch.nn.functional.interpolate(x[None, None], scale_factor=1.0 / scale,
                                            mode="trilinear", align_corners=False)[0, 0]
        ds_shape = tuple(x.shape)  # scaled dims before padding
        pad = [max(PATCH - s, 0) for s in x.shape]
        if any(pad):  # scaled crop smaller than the window: zero-pad (masked air is 0)
            x = torch.nn.functional.pad(x, (0, pad[2], 0, pad[1], 0, pad[0]))
    D, H, W = x.shape
    if half:
        x = x.half()
    prob = torch.zeros((D, H, W), device=DEV)
    wsum = torch.zeros((D, H, W), device=DEV)
    w = gauss3(PATCH).to(DEV)
    zs = sorted(set(list(range(0, max(D - PATCH, 0) + 1, STRIDE)) + [max(D - PATCH, 0)]))
    wins = [(z0, y0, x0) for z0 in zs for y0 in zs for x0 in zs]
    for i in range(0, len(wins), wbatch):
        grp = wins[i:i + wbatch]
        p = torch.stack([x[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] for z0, y0, x0 in grp])[:, None]
        if half:
            p = p.to(memory_format=torch.channels_last_3d)
            lo = net(p)
        else:
            with torch.autocast("cuda", torch.bfloat16):
                lo = net(p)
        pr = torch.softmax(lo.float(), 1)[:, 1]
        for j, (z0, y0, x0) in enumerate(grp):
            prob[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] += pr[j] * w
            wsum[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH] += w
    out = prob / wsum.clamp_min(1e-6)
    if scale > 1:
        out = out[:ds_shape[0], :ds_shape[1], :ds_shape[2]]
        out = torch.nn.functional.interpolate(out[None, None], size=full_shape,
                                              mode="trilinear", align_corners=False)[0, 0]
    return out.cpu().numpy()


def surf_pts(mask):
    return mask & ~ndimage.binary_erosion(mask)


def surface_dice(a, b, tau=2.0):
    if not a.any() or not b.any():
        return 0.0 if (a.any() != b.any()) else 1.0
    sa, sb = surf_pts(a), surf_pts(b)
    da = ndimage.distance_transform_edt(~sa)
    db = ndimage.distance_transform_edt(~sb)
    return float((db[sa] <= tau).sum() + (da[sb] <= tau).sum()) / float(sa.sum() + sb.sum())


def auprc_maxdice(prob, sheet, lab):
    """Threshold-free ranking metrics over labeled voxels: exact average precision and
    the max dice along the PR curve (+ the probability threshold achieving it)."""
    p = prob[lab].ravel()
    g = sheet[lab].ravel()
    o = np.argsort(-p)
    gs = g[o]
    tp = np.cumsum(gs, dtype=np.float64)
    fp = np.cumsum(~gs, dtype=np.float64)
    npos = float(gs.sum())
    if npos == 0:
        return 0.0, 0.0, 0.5
    rec = tp / npos
    prec = tp / (tp + fp)
    ap = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    dice = 2 * tp / (tp + fp + npos)
    i = int(np.argmax(dice))
    return ap, float(dice[i]), float(p[o][i])


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
    ap.add_argument("--scale", type=int, default=1,
                    help="coarse-canon models (canon=2.4*scale): downsample CT before predict, "
                         "upsample probs back; GT stays 2.4um")
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
        sheet = gt >= 192
        lab = gt >= 64
        if sheet.sum() < 1000:
            print(f"{cd}: skipping (only {sheet.sum()} sheet voxels)")
            continue
        r = {"crop": os.path.basename(cd), "n_sheet": int(sheet.sum()), "n_lab": int(lab.sum())}
        for name, net in nets.items():
            prob = predict(net, ct, scale=args.scale)
            pred = (prob > 0.5) & lab
            inter = (pred & sheet).sum()
            r[f"dice_{name}"] = float(2 * inter / max(pred.sum() + sheet.sum(), 1))
            ap, md, mt = auprc_maxdice(prob, sheet, lab)
            r[f"auprc_{name}"], r[f"maxdice_{name}"], r[f"maxdice_t_{name}"] = ap, md, mt
            r[f"sd2_{name}"] = surface_dice((prob > mt) & lab, sheet, 2.0)
            r[f"ece_{name}"] = ece(prob, sheet, lab)
        rows.append(r)
        print(f"{r['crop']}: " + " | ".join(
            f"{n} dice {r[f'dice_{n}']:.3f} maxdice {r[f'maxdice_{n}']:.3f}@{r[f'maxdice_t_{n}']:.3f} "
            f"auprc {r[f'auprc_{n}']:.3f} sd2 {r[f'sd2_{n}']:.3f} ece {r[f'ece_{n}']:.3f}"
            for n in nets), flush=True)

    summary = {}
    for name in nets:
        for m in ("dice", "maxdice", "auprc", "sd2", "ece"):
            v = np.array([r[f"{m}_{name}"] for r in rows])
            summary[f"{m}_{name}"] = {"mean": float(v.mean()), "p10": float(np.percentile(v, 10)),
                                      "min": float(v.min())}
            print(f"{name} {m}: mean {v.mean():.4f}  p10 {np.percentile(v,10):.4f}  min {v.min():.4f}")
    json.dump({"crops": rows, "summary": summary}, open(args.out, "w"), indent=1)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
