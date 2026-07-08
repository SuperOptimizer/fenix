"""Separable-student KD experiment: 3x3x3 convs -> three 1D convs (27 -> 9 taps, 3x
FLOPs), R(2+1)D-style, channel-preserving.

Warm start: each 1D kernel is the teacher kernel collapsed along the other two axes
(exact for rank-1-separable kernels; a reasonable init otherwise). Then short KD
self-distillation on real CT — the question is loss-recovery TREND vs a dense student
at the same step count. Runs the student in fp16-autocast (our custom kernels are
cubic-only today; int8/fp8 separable support = follow-up if the accuracy verdict is
good, k=(3,1,1) is just a rectangular tap table).

Usage: CUDA_VISIBLE_DEVICES=0 python sep_student.py [--steps 60]
"""
import argparse
import copy
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F


class SepConv3d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        Ci, Co = conv.in_channels, conv.out_channels
        s = conv.stride
        self.a = torch.nn.Conv3d(Ci, Co, (3, 1, 1), stride=(s[0], 1, 1),
                                 padding=(1, 0, 0), bias=False)
        self.b = torch.nn.Conv3d(Co, Co, (1, 3, 1), stride=(1, s[1], 1),
                                 padding=(0, 1, 0), bias=False)
        self.c = torch.nn.Conv3d(Co, Co, (1, 1, 3), stride=(1, 1, s[2]),
                                 padding=(0, 0, 1), bias=conv.bias is not None)
        with torch.no_grad():
            w = conv.weight.detach()                      # [Co,Ci,3,3,3]
            self.a.weight.copy_(w.mean((3, 4), keepdim=True))
            eye = torch.zeros_like(self.b.weight)
            for co in range(Co):
                eye[co, co, 0, :, 0] = w[co].mean((0, 1, 3)) / \
                    w[co].mean((0, 1, 3)).abs().sum().clamp(min=1e-6)
            self.b.weight.copy_(eye)
            eye2 = torch.zeros_like(self.c.weight)
            for co in range(Co):
                eye2[co, co, 0, 0, :] = w[co].mean((0, 1, 2)) / \
                    w[co].mean((0, 1, 2)).abs().sum().clamp(min=1e-6)
            self.c.weight.copy_(eye2)
            if conv.bias is not None:
                self.c.bias.copy_(conv.bias.detach())

    def forward(self, x):
        return self.c(self.b(self.a(x)))


def separate(net):
    n = 0
    for name, mod in list(net.named_modules()):
        if "squeeze_excitation" in name:
            continue
        for cname, child in list(mod.named_children()):
            if (isinstance(child, torch.nn.Conv3d) and child.kernel_size == (3, 3, 3)
                    and child.groups == 1 and child.dilation == (1, 1, 1)):
                setattr(mod, cname, SepConv3d(child))
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    from reference import build_and_load
    ckpt = ("/home/forrest/fenix/models/surface_recto_3dunet/"
            "checkpoint_inference_ready.pth")
    teacher = build_and_load(ckpt).to(dev).eval()
    sep = copy.deepcopy(teacher)
    ns = separate(sep)
    sep = sep.to(dev)   # SepConv3d replacements are built on CPU
    dense = copy.deepcopy(teacher)   # dense control at same lr/steps (perturbed)
    g = torch.Generator(device=dev).manual_seed(7)
    with torch.no_grad():
        for p in dense.parameters():
            if p.dim() >= 2:
                p.add_(torch.randn(p.shape, generator=g, device=dev) * (0.03 * p.std()))
    p_t = sum(p.numel() for p in teacher.parameters())
    p_s = sum(p.numel() for p in sep.parameters())
    print(f"separated {ns} convs; params {p_t/1e6:.1f}M -> {p_s/1e6:.1f}M")
    sep.train()
    dense.train()
    teacher_cl = teacher.to(memory_format=torch.channels_last_3d)

    vol = np.load("/tmp/ct/patch.npy")
    opt_s = torch.optim.Adam(sep.parameters(), lr=args.lr)
    opt_d = torch.optim.Adam(dense.parameters(), lr=args.lr)
    scaler_s, scaler_d = torch.amp.GradScaler("cuda"), torch.amp.GradScaler("cuda")

    def kl(logits, t):
        nvox = logits.shape[2] * logits.shape[3] * logits.shape[4]
        return F.kl_div(logits.float().log_softmax(1), t, reduction="sum") / nvox

    rng = np.random.default_rng(0)
    P = 128
    ts, td = [], []
    for i in range(args.steps):
        z, y, xx = (int(rng.integers(0, 256 - P)) for _ in range(3))
        c = torch.from_numpy(vol[z:z + P, y:y + P, xx:xx + P]).float()
        x = ((c - c.mean()) / c.std().clamp(min=1e-6))[None, None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            t = teacher_cl(x.to(memory_format=torch.channels_last_3d)).float().softmax(1)
        for net, opt, sc, acc in ((sep, opt_s, scaler_s, ts),
                                  (dense, opt_d, scaler_d, td)):
            opt.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(x)
            loss = kl(logits, t)
            sc.scale(loss).backward()
            sc.step(opt)
            sc.update()
            torch.cuda.synchronize()
            acc.append((loss.item(), time.perf_counter() - t0))
        if i % max(1, args.steps // 10) == 0 or i == args.steps - 1:
            print(f"  step {i:>3}: sep loss {ts[-1][0]:.5f} ({ts[-1][1]*1e3:5.0f}ms) | "
                  f"dense loss {td[-1][0]:.5f} ({td[-1][1]*1e3:5.0f}ms)")
    last = args.steps // 4
    print(f"\nlast-quarter mean loss: sep {statistics.mean(l for l, _ in ts[-last:]):.5f}"
          f" vs dense {statistics.mean(l for l, _ in td[-last:]):.5f} | median step "
          f"{statistics.median(t for _, t in ts)*1e3:.0f} vs "
          f"{statistics.median(t for _, t in td)*1e3:.0f} ms")


if __name__ == "__main__":
    main()
