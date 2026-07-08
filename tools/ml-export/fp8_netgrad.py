"""Whole-net fp8 gradient check on real CT (sm120).

Builds the surface ResEnc-UNet twice, swaps every eligible conv3d in one copy to the
fp8 fwd+bwd path (Fp8Conv3d autograd Function), runs the SAME fwd+bwd through both,
and compares per-parameter gradients against full-precision autograd. Validates that
the fp8 training path is correct end to end across all 252 convs in composition —
not just per-op (fp8_train.py covers that).

Usage: python fp8_netgrad.py [--patch 128] [--ct /tmp/ct/patch.npy]
Patch must be divisible by 64 (7-stage net). Expects the real-CT patch from
fp8_forward.py / `fenix ingest-zarr` — synthetic noise is OOD and misleading here.
"""
import argparse
import copy
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import fp8_forward as ff
from fp8_conv3d_op import dump_tuned, load_tuned
from fp8_train import swap_convs_fp8
from reference import build_and_load


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--ct", default="/tmp/ct/patch.npy")
    ap.add_argument("--ckpt", default="/home/forrest/fenix/models/surface_recto_3dunet/"
                                      "checkpoint_inference_ready.pth")
    ap.add_argument("--fused", action="store_true",
                    help="use the P2.2 fused CDNR blocks (fp8-resident) instead of "
                         "per-conv swapping")
    ap.add_argument("--normfuse", action="store_true",
                    help="also swap InstanceNorm+LeakyReLU for the layout-native "
                         "fp8 norm+act (composes with the default per-conv swap)")
    ap.add_argument("--tailfuse", action="store_true",
                    help="also fuse BasicBlockD tails (scSE+residual+lrelu)")
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)

    tuned = os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json")
    if os.path.exists(tuned):
        print(f"pinned autotune configs: {load_tuned(tuned)}")

    net = build_and_load(args.ckpt).to(dev)
    net8 = copy.deepcopy(net)
    if args.fused:
        from fp8_train_fused import swap_cdnr_fp8
        nf, ns, nk = swap_cdnr_fp8(net8)
        print(f"fused {nf} CDNR blocks + {ns} bare convs to fp8, kept {nk}")
    else:
        ns, nk = swap_convs_fp8(net8)
        nn_ = nt = 0
        if args.normfuse:
            from fp8_train import swap_norms_fp8
            nn_ = swap_norms_fp8(net8)
        if args.tailfuse:
            from fp8_train import swap_tails_fp8
            nt = swap_tails_fp8(net8)
        print(f"swapped {ns} convs to fp8, kept {nk}; {nn_} norm+act, {nt} tails fused")

    x = ff.load_ct_patch(args.ct, args.patch, dev)
    net.train()
    net8.train()
    y = net(x)
    y8 = net8(x)
    print("fwd logit corr:", round(corr(y8, y), 5))

    gy = torch.randn_like(y) * 0.1
    y.backward(gy)
    y8.backward(gy.to(y8.dtype))

    rows = []
    ndead = 0
    p8map = dict(net8.named_parameters())
    for n, p in net.named_parameters():
        n8 = n
        if n8 not in p8map:     # fused layers flatten names (conv.weight -> weight, ...)
            for a, b in ((".conv.weight", ".weight"), (".conv.bias", ".bias"),
                         (".norm.weight", ".gamma"), (".norm.bias", ".beta")):
                if n.endswith(a) and n[:-len(a)] + b in p8map:
                    n8 = n[:-len(a)] + b
                    break
        p8 = p8map.get(n8)
        if p8 is None or p.grad is None or p8.grad is None or p.grad.numel() <= 100:
            continue
        c = corr(p8.grad, p.grad)
        if math.isnan(c):       # dead params (zero grad in BOTH nets) poison the median
            ndead += 1
            continue
        rows.append((n, c))
    rows.sort(key=lambda r: r[1])
    print("worst 5 param-grad corrs:")
    for n, c in rows[:5]:
        print(f"  {c:.4f} {n[:70]}")
    med = statistics.median([c for _, c in rows])
    wrows = [c for n, c in rows if not n.endswith(".bias")]
    print(f"median grad corr: {med:.4f} over {len(rows)} params "
          f"({ndead} dead excluded); weights-only median: "
          f"{statistics.median(wrows):.4f} over {len(wrows)}")
    print(f"pinned {dump_tuned(tuned)} tuned configs -> {tuned}")


if __name__ == "__main__":
    main()
