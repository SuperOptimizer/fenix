"""Export the surface ResEnc-UNet as an AOTInductor `.pt2` package for the C++ predict path.

The package bakes torch.compile's fused kernels (norm/act/scSE fused around the cuDNN/MIOpen
convs — measured 1.8x over eager on MI300X, 1.3x on 5060 Ti) into a self-contained artifact
that `fenix predict-surface` / `predict-scroll` load via AotiNet (src/ml/aoti_net.hpp), with
no Python at inference time. Packages are STATIC-shape (batch/patch fixed here) and
torch-version/GPU-arch locked: export ON the box that will run it (project no-compat rule).

The net is exported in pure fp16 (weights + activations; input/output fp16) — the same
numerics infer.hpp's opt.half path feeds it. A validation forward against the eager
autocast reference is printed at the end; expect corr > 0.999.

Usage:
  python export_aoti.py <checkpoint.pth> <out.pt2> [--batch 3] [--patch 256]
                        [--no-max-autotune]
"""
import argparse
import sys

import torch

from reference import build_and_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out", help="output package path (.pt2)")
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--no-max-autotune", action="store_true",
                    help="skip autotuning (much faster export, slightly slower kernels)")
    ap.add_argument("--ink", action="store_true",
                    help="export the ink_3d_dino_guided net (ema_model key, no scSE) instead of surface")
    args = ap.parse_args()
    if not args.out.endswith(".pt2"):
        sys.exit("out must end with .pt2 (the C++ dispatch keys on the suffix)")

    dev = "cuda"
    net = build_and_load(args.ckpt, ink=args.ink).to(dev).eval().half()
    B, P = args.batch, args.patch
    x = torch.randn(B, 1, P, P, P, device=dev, dtype=torch.float16)

    with torch.no_grad():
        ncls = net(x).shape[1]
        ep = torch.export.export(net, (x,))
        cfg = {
            "aot_inductor.metadata": {
                "fenix_batch": str(B),
                "fenix_patch": str(P),
                "fenix_classes": str(ncls),
            },
        }
        if not args.no_max_autotune:
            cfg["max_autotune"] = True
            cfg["coordinate_descent_tuning"] = True
        pkg = torch._inductor.aoti_compile_and_package(
            ep, package_path=args.out, inductor_configs=cfg)
    print(f"packaged: {pkg}  (batch={B} patch={P} classes={ncls})", flush=True)

    # Validate: package vs eager-autocast reference (the current production numerics).
    ref = build_and_load(args.ckpt, ink=args.ink).to(dev).eval()
    loaded = torch._inductor.aoti_load_package(args.out)
    xv = torch.randn(1, 1, P, P, P, device=dev, dtype=torch.float32)
    def prob(y):
        return y.float().sigmoid()[:, 0] if args.ink else y.float().softmax(1)[:, 1]
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            yr = prob(ref(xv))
        xb = torch.zeros(B, 1, P, P, P, device=dev, dtype=torch.float16)
        xb[0].copy_(xv[0].half())
        yp = prob(loaded(xb)[0:1])
    d = (yr - yp).abs()
    corr = torch.corrcoef(torch.stack([yr.flatten(), yp.flatten()]))[0, 1].item()
    print(f"validate vs eager-autocast: max|dP|={d.max().item():.4g} "
          f"mean|dP|={d.mean().item():.4g} corr={corr:.6f}", flush=True)
    if corr < 0.999:
        print("WARNING: corr < 0.999 — inspect before production use", flush=True)


if __name__ == "__main__":
    main()
