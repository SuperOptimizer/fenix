"""FullScrollNet training smoke on phantoms: prove every head learns.

Generates a small pool of synthetic scroll phantoms (dense GT for all four heads),
trains fp16-autocast, and reports per-head losses + metrics. Success bar for the
smoke: papyrus Dice > 0.9, circular-w MAE < 0.05, normal error < 15 deg, ink Dice
clearly climbing — on TRAIN-pool phantoms this is a capacity/plumbing check, plus
a small held-out pool for a generalization glance.

  CUDA_VISIBLE_DEVICES=0 python train_smoke.py [--steps 200] [--pool 12] [--batch 2]
      [--warm /path/to/surface_ckpt.pth] [--size 128]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from losses import full_scroll_loss, head_metrics
from model import FullScrollNet, warm_start
from phantom import make_phantom, to_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--pool", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warm", default="")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    net = FullScrollNet().to(dev)
    if args.warm:
        loaded, total = warm_start(net, args.warm)
        print(f"warm-start: {loaded/1e6:.1f}M/{total/1e6:.1f}M params from surface ckpt")
    n_par = sum(p.numel() for p in net.parameters())
    print(f"FullScrollNet: {n_par/1e6:.1f}M params, heads sem/normal/wind/ink")

    t0 = time.time()
    pool = [make_phantom(args.size, seed=s) for s in range(args.pool)]
    held = [make_phantom(args.size, seed=1000 + s) for s in range(2)]
    print(f"phantoms: {args.pool}+2 x {args.size}^3 in {time.time()-t0:.0f}s")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")
    net.train()
    import numpy as np
    rng = np.random.default_rng(0)
    step_ms = []
    for i in range(args.steps):
        idx = rng.choice(args.pool, args.batch, replace=False)
        x, t = to_batch([pool[j] for j in idx], dev)
        opt.zero_grad(set_to_none=True)
        ts = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
        loss, parts = full_scroll_loss(out, t)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - ts) * 1e3)
        if i % max(1, args.steps // 10) == 0 or i == args.steps - 1:
            p = " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
            print(f"  step {i:>4}: total={loss.item():.4f} | {p} "
                  f"({step_ms[-1]:.0f}ms)")

    net.eval()
    for tag, batch in (("train-pool", pool[:2]), ("held-out", held)):
        x, t = to_batch(batch, dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            out = net(x)
        m = head_metrics({k: v.float() for k, v in out.items()}, t)
        print(f"{tag}: " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    import statistics
    print(f"median step {statistics.median(step_ms):.0f}ms | batch {args.batch} "
          f"x {args.size}^3")


if __name__ == "__main__":
    main()
