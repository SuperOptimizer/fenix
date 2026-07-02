#!/usr/bin/env python3
"""train.py — student KD training loop (docs/design/training-pipeline.md).

Consumes the `fenix train-feed` shm ring (CT, GT band[, teacher prob] u8 patches),
trains a shrunk ResEnc-UNet student in bf16 autocast with
  loss = alpha * KL(student || teacher, T=2)  +  beta * (Dice + CE)(student, GT band)
EMA weights for eval/export; torchao int8 QAT switchable for the final phase.

Usage (box):
  fenix train-feed pairs.txt /dev/shm/feed.ring patch=256 slots=32 threads=16 &
  trainenv/bin/python tools/train/train.py --ring /dev/shm/feed.ring --steps 100000 \
      --batch 8 --out /workspace/student

This is the phase-3 skeleton: arch config + LR schedule sweeps come once the bulk-KD
artifacts exist. Every mechanism (ring IO, bf16, KD loss, EMA, checkpoint/resume, QAT
hook) is real and runnable now.
"""
import argparse
import copy
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from feed_reader import FeedRing


# ---- student: a shrunk ResEnc-UNet (teacher arch family, fewer stages/filters) ---------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.n1 = nn.InstanceNorm3d(cout, affine=True)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = nn.InstanceNorm3d(cout, affine=True)
        self.skip = (nn.Conv3d(cin, cout, 1, stride=stride, bias=False)
                     if (cin != cout or stride != 1) else nn.Identity())

    def forward(self, x):
        h = F.leaky_relu(self.n1(self.c1(x)), 0.01)
        h = self.n2(self.c2(h))
        return F.leaky_relu(h + self.skip(x), 0.01)


class StudentUNet(nn.Module):
    """4-stage encoder/decoder, ~1/4 the teacher's width. Fully convolutional."""

    def __init__(self, base=16, classes=2):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e0 = ResBlock(1, c[0])
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

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        d0 = self.d0(torch.cat([self.u0(d1), e0], 1))
        return self.head(d0)


def dice_loss(logits, target):
    p = torch.softmax(logits, 1)[:, 1]
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + 1) / (denom + 1)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.7, help="KD weight (0 if no teacher channel)")
    ap.add_argument("--beta", type=float, default=0.3, help="GT-band weight")
    ap.add_argument("--kd-T", type=float, default=2.0)
    ap.add_argument("--base", type=int, default=16, help="student base width")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out", default="student")
    ap.add_argument("--resume", default="")
    ap.add_argument("--qat", action="store_true", help="enable torchao int8 QAT (final phase)")
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    net = StudentUNet(base=args.base).to(dev)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    step0 = 0

    if args.resume and os.path.exists(args.resume):
        st = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(st["net"])
        ema.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step0 = st["step"]
        print(f"resumed from {args.resume} at step {step0}")

    if args.qat:
        from torchao.quantization.qat import IntXQuantizationAwareTrainingConfig, quantize_
        quantize_(net, IntXQuantizationAwareTrainingConfig())
        print("torchao int8 QAT enabled")

    ring = FeedRing(args.ring)
    print(f"ring: {ring.nslots} slots, patch={ring.patch}, channels={ring.channels}")
    has_teacher = ring.channels == 3
    if not has_teacher and args.alpha > 0:
        print("no teacher channel in ring -> alpha forced to 0 (GT-only training)")
        args.alpha = 0.0

    t0, seen = time.time(), 0
    for step in range(step0, args.steps):
        b = ring.next_batch(args.batch)
        ct = torch.from_numpy(b["ct"]).to(dev, non_blocking=True)
        gt = torch.from_numpy(b["gt"]).to(dev, non_blocking=True)
        x = ct.unsqueeze(1).float()
        x = (x - x.mean(dim=(2, 3, 4), keepdim=True)) / (x.std(dim=(2, 3, 4), keepdim=True) + 1e-6)
        y = (gt > 127).float()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
            loss = args.beta * (dice_loss(logits.float(), y) +
                                F.cross_entropy(logits.float(), y.long()))
            if args.alpha > 0:
                tprob = (torch.from_numpy(b["teacher"]).to(dev).float() / 255.0).clamp(1e-4, 1 - 1e-4)
                T = args.kd_T
                slog = F.log_softmax(logits.float() / T, dim=1)[:, 1]
                # binary KL against the teacher's saturated prob, temperature-softened
                tsoft = ((tprob.logit()) / T).sigmoid()
                kd = F.binary_cross_entropy_with_logits(
                    (logits.float()[:, 1] - logits.float()[:, 0]) / T, tsoft) * (T * T)
                loss = loss + args.alpha * kd
                del slog

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - args.ema)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)

        seen += args.batch
        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step}  loss {loss.item():.4f}  {seen / max(dt, 1e-9):.1f} patches/s  "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            path = f"{args.out}_step{step}.pt"
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step},
                       path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"checkpoint -> {path}", flush=True)

    torch.save({"ema": ema.state_dict(), "step": args.steps}, f"{args.out}_final.pt")
    print("done")


if __name__ == "__main__":
    main()
