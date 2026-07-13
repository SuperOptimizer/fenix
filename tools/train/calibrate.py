#!/usr/bin/env python3
"""Temperature-scale a student checkpoint (the studentH ECE-0.445 wart fixer).

Fits ONE softmax temperature T by binary NLL (LBFGS) on labeled voxels from the
ODD-indexed holdout crops, reports ECE before/after on the EVEN-indexed crops
(never seen by the fit), then BAKES T into the head conv (weight,bias /= T ==
softmax(z/T) exactly, since the head is a 1x1 Conv3d) so the output stays a plain
StudentUNet checkpoint — .ts export, TRT, trace_eval_run all work unchanged.
Ranking metrics (auprc/maxdice/trace recall) are monotone-invariant under T; only
probability/threshold consumers change. Verify end-to-end with eval_students.py
on the emitted checkpoint.

Usage:
  calibrate.py --ckpt /tmp/gtqc/real/studentH_best.pt [--base 16]
               [--crops /tmp/gtqc/m8/eval] --out /tmp/gtqc/real/studentH_best_cal.pt
"""
import os, sys, glob, json, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import StudentUNet
from eval_students import ece

DEV = "cuda:0"
PATCH = 128


@torch.no_grad()
def crop_dlogits(net, ct, gt, cap=500_000, rng=None):
    """Non-overlapping windows; returns (dlogit, y) at labeled voxels, subsampled."""
    D, H, W = ct.shape
    x = torch.from_numpy(ct.astype(np.float32) / 255.0).to(DEV)
    zs = sorted(set(list(range(0, max(D - PATCH, 0) + 1, PATCH)) + [max(D - PATCH, 0)]))
    ds, ys = [], []
    for z0 in zs:
        for y0 in zs:
            for x0 in zs:
                g = gt[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH]
                lab = g >= 64
                if lab.sum() < 1000:
                    continue
                p = x[z0:z0+PATCH, y0:y0+PATCH, x0:x0+PATCH][None, None]
                with torch.autocast("cuda", torch.bfloat16):
                    lo = net(p)
                d = (lo[0, 1] - lo[0, 0]).float().cpu().numpy()
                ds.append(d[lab]); ys.append((g >= 192)[lab])
    d = np.concatenate(ds); y = np.concatenate(ys)
    if len(d) > cap:
        idx = (rng or np.random.default_rng(0)).choice(len(d), cap, replace=False)
        d, y = d[idx], y[idx]
    return d, y


def fit_temperature(d, y):
    dt = torch.from_numpy(d).double().to(DEV)
    yt = torch.from_numpy(y.astype(np.float64)).to(DEV)
    log_t = torch.zeros(1, dtype=torch.float64, device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=80)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(dt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--crops", default="/tmp/gtqc/m8/eval")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    st = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = st.get("ema", st.get("net")) if isinstance(st, dict) else st
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    net = StudentUNet(base=args.base)
    net.load_state_dict(sd)
    net = net.to(DEV).eval()

    crops = sorted(glob.glob(f"{args.crops}/crop*"))
    fit_c, val_c = crops[1::2], crops[0::2]
    rng = np.random.default_rng(0)

    def collect(cs):
        ds, ys = [], []
        for cd in cs:
            ct = np.load(f"{cd}/ct.npy"); gt = np.load(f"{cd}/gt.npy")
            d, y = crop_dlogits(net, ct, gt, rng=rng)
            ds.append(d); ys.append(y)
            print(f"  {os.path.basename(cd)}: {len(d)} voxels", flush=True)
        return np.concatenate(ds), np.concatenate(ys)

    print(f"fit crops ({len(fit_c)}):")
    df, yf = collect(fit_c)
    T = fit_temperature(df, yf)
    print(f"temperature T = {T:.3f}")

    print(f"val crops ({len(val_c)}):")
    dv, yv = collect(val_c)
    def ece_at(d, y, t):
        return ece(1 / (1 + np.exp(-d / t)), y, np.ones(len(d), bool))

    e0, e1 = ece_at(dv, yv, 1.0), ece_at(dv, yv, T)
    print(f"val ECE: {e0:.4f} -> {e1:.4f}  (fit-side {ece_at(df, yf, 1.0):.4f} -> {ece_at(df, yf, T):.4f})")

    for k in ("head.weight", "head.bias"):
        sd[k] = sd[k] / T
    torch.save({"ema": sd}, args.out)  # load_student expects the ema/net wrapper
    json.dump({"ckpt": args.ckpt, "T": T, "val_ece_before": e0, "val_ece_after": e1,
               "fit_crops": [os.path.basename(c) for c in fit_c]},
              open(os.path.splitext(args.out)[0] + ".json", "w"), indent=1)
    print(f"-> {args.out} (T baked into head)")


if __name__ == "__main__":
    main()
