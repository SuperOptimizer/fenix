#!/usr/bin/env python3
"""M8 ablation eval: studentG (graded accept-set) vs studentR (raw corpus) on 12 held-out
crops (wraps w081-084 + w089-091, excluded from both training arms).

GT: multi-mesh raster (255 sheet / 128 trusted-bg / 0 ignore). Metrics on labeled voxels:
dice on the sheet band + surface-dice@2 (band surfaces, scipy EDT). Inference: sliding
window 128, 50% overlap, Gaussian blend, no TTA (identical for both arms).
"""
import sys, os, glob, json
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import ndimage

sys.path.insert(0, "/home/forrest/fenix/tools/train")
from train import StudentUNet

DEV = "cuda:0"
PATCH, STRIDE = 128, 64


def load_student(ckpt):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = st.get("ema", st.get("net"))
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    net = StudentUNet(base=16)
    net.load_state_dict(sd)
    return net.to(DEV).eval()


def gauss3(n, sigma_frac=0.25):
    ax = np.arange(n) - (n - 1) / 2.0
    g = np.exp(-(ax ** 2) / (2 * (n * sigma_frac) ** 2)).astype(np.float32)
    w = g[:, None, None] * g[None, :, None] * g[None, None, :]
    return torch.from_numpy(w)


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
    er = ndimage.binary_erosion(mask)
    return mask & ~er


def surface_dice(a, b, tau=2.0):
    if not a.any() or not b.any():
        return 0.0 if (a.any() != b.any()) else 1.0
    sa, sb = surf_pts(a), surf_pts(b)
    da = ndimage.distance_transform_edt(~sa)
    db = ndimage.distance_transform_edt(~sb)
    na, nb = sa.sum(), sb.sum()
    return float((db[sa] <= tau).sum() + (da[sb] <= tau).sum()) / float(na + nb)


def main():
    nets = {arm: load_student(ck) for arm, ck in
            [("G", sys.argv[1]), ("R", sys.argv[2])]}
    rows = []
    for cd in sorted(glob.glob("/tmp/gtqc/m8/eval/crop*")):
        ct = np.load(f"{cd}/ct.npy")
        gt = np.load(f"{cd}/gt.npy")
        lab = gt > 0                      # labeled voxels only
        sheet = gt == 255
        if sheet.sum() < 1000:
            print(f"{cd}: skipping (only {sheet.sum()} sheet voxels)")
            continue
        r = {"crop": os.path.basename(cd), "n_sheet": int(sheet.sum()), "n_lab": int(lab.sum())}
        for arm, net in nets.items():
            prob = predict(net, ct)
            pred = (prob > 0.5) & lab     # ignore-region masked out
            inter = (pred & sheet).sum()
            r[f"dice_{arm}"] = float(2 * inter / max(pred.sum() + sheet.sum(), 1))
            r[f"sd2_{arm}"] = surface_dice(pred, sheet & lab, 2.0)
        rows.append(r)
        print(f"{r['crop']}: dice G {r['dice_G']:.3f} R {r['dice_R']:.3f} | "
              f"sd2 G {r['sd2_G']:.3f} R {r['sd2_R']:.3f}", flush=True)
    for m in ("dice", "sd2"):
        g = np.mean([r[f"{m}_G"] for r in rows]); rr = np.mean([r[f"{m}_R"] for r in rows])
        print(f"MEAN {m}: graded {g:.4f}  raw {rr:.4f}  delta {g-rr:+.4f}")
    json.dump(rows, open("/tmp/gtqc/m8/eval_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
