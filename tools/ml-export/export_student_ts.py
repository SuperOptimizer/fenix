#!/usr/bin/env python3
"""export_student_ts.py — package a distilled StudentUNet checkpoint (EMA weights from
finetune_ink_compression.py --student-base) as a TorchScript .ts that fenix's JitNet
path loads directly (predict-surface / predict-scroll <weights>.ts).

Output contract matches the existing C++ per-net plumbing with ZERO C++ changes:
  --task ink      -> raw 1-channel sheet logit      (net=ink: sigmoid, channel 0)
  --task surface  -> cat([0, logit]) 2-channel      (net=surface: softmax, channel 1;
                     softmax([0, y])[1] == sigmoid(y), exactly)

  python3 export_student_ts.py --ckpt surf_student_b64.pth --task surface \
      --base 64 --stem 2 --out surf_student_b64.ts
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from train import ResBlock, StudentUNet  # noqa: E402


class ExportStudent(nn.Module):
    # script-clean twin of StudentUNet (same attribute names -> strict state-dict load).
    # StudentUNet.forward's aux flag returns Tensor-or-tuple, which TorchScript rejects,
    # and jit.script always compiles a submodule's forward — so the twin, not a wrapper.
    def __init__(self, base: int, classes: int, stem_stride: int, two_channel: bool):
        super().__init__()
        self.stem_stride = stem_stride
        self.two_channel = two_channel
        c = [base, base * 2, base * 4, base * 8]
        self.e0 = ResBlock(1, c[0], stride=stem_stride)
        self.e1 = ResBlock(c[0], c[1], stride=2)
        self.e2 = ResBlock(c[1], c[2], stride=2)
        self.e3 = ResBlock(c[2], c[3], stride=2)
        self.u2 = nn.ConvTranspose3d(c[3], c[2], 2, stride=2)
        self.d2 = ResBlock(c[2] * 2, c[2])
        self.u1 = nn.ConvTranspose3d(c[2], c[1], 2, stride=2)
        self.d1 = ResBlock(c[1] * 2, c[1])
        self.u0 = nn.ConvTranspose3d(c[1], c[0], 2, stride=2)
        self.d0 = ResBlock(c[0] * 2, c[0])
        self.head = nn.Conv3d(c[0], classes, 1)
        self.aux_head = nn.Conv3d(c[0], 2, 1)  # unused; present so strict load passes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        y = self.head(self.d0(torch.cat([self.u0(d1), e0], 1)))
        if self.stem_stride > 1:
            y = F.interpolate(y, size=x.shape[2:], mode="trilinear", align_corners=False)
        if self.two_channel:
            y = torch.cat([torch.zeros_like(y), y], dim=1)
        return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["ink", "surface"], required=True)
    ap.add_argument("--base", type=int, default=64)
    ap.add_argument("--stem", type=int, default=2)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    wrap = ExportStudent(args.base, 1, args.stem, two_channel=(args.task == "surface")).eval()
    wrap.load_state_dict(ck["ema_model"])
    ref = StudentUNet(base=args.base, classes=1, stem_stride=args.stem).eval()
    ref.load_state_dict(ck["ema_model"])
    n_par = sum(p.numel() for p in wrap.parameters())
    print(f"loaded {args.ckpt} step {ck.get('step')} ({n_par/1e6:.1f}M params)", flush=True)

    mod = torch.jit.script(wrap)
    mod.save(args.out)
    print(f"saved {args.out}", flush=True)

    # parity vs the ORIGINAL training class (catches twin drift) + dynamic-shape check
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ref, mod = ref.to(dev), torch.jit.load(args.out, map_location=dev).eval()
    for s in (128, 96):
        x = torch.randn(1, 1, s, s, s, device=dev)
        with torch.no_grad():
            a, b = ref(x), mod(x)
        assert b.shape == (1, 2 if args.task == "surface" else 1, s, s, s), b.shape
        logit = b[:, 1:] if args.task == "surface" else b
        d = (a - logit).abs().max().item()
        if args.task == "surface":
            d = max(d, (torch.sigmoid(a[:, 0]) - torch.softmax(b, 1)[:, 1]).abs().max().item())
        print(f"parity {s}^3: max |train-class eager - ts| {d:.2e}", flush=True)
        assert d < 1e-5, d
    print("EXPORT_OK", flush=True)


if __name__ == "__main__":
    main()
