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
import json
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


def dice_loss(logits, target, mask):
    p = torch.softmax(logits, 1)[:, 1] * mask
    t = target * mask
    inter = (p * t).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + t.sum(dim=(1, 2, 3))
    return (1 - (2 * inter + 1) / (denom + 1)).mean()


# soft-clDice (Shit et al., CVPR 2021): connectivity-aware loss for thin structures. Dice+CE
# don't punish sheet-specific failures (micro-holes, cross-sheet bridges); clDice compares
# soft SKELETONS (iterative min/max-pool morphology — for sheets the medial surface) so a
# 1-voxel hole in a sheet costs as much as a blob of the same area. eval/ computes the exact
# metric; this is its differentiable train-time counterpart.
def _soft_erode(img):
    return -F.max_pool3d(-img, 3, 1, 1)


def _soft_dilate(img):
    return F.max_pool3d(img, 3, 1, 1)


def soft_skel(img, iters):
    img1 = _soft_dilate(_soft_erode(img))
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_dilate(_soft_erode(img))
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cldice_loss(logits, target, mask, iters=3):
    p = (torch.softmax(logits, 1)[:, 1] * mask).unsqueeze(1)
    t = (target * mask).unsqueeze(1)
    sp, st = soft_skel(p, iters), soft_skel(t, iters)
    tprec = ((sp * t).sum(dim=(1, 2, 3, 4)) + 1) / (sp.sum(dim=(1, 2, 3, 4)) + 1)
    tsens = ((st * p).sum(dim=(1, 2, 3, 4)) + 1) / (st.sum(dim=(1, 2, 3, 4)) + 1)
    return (1 - 2 * tprec * tsens / (tprec + tsens)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ring", required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.7, help="KD weight (0 if no teacher channel)")
    ap.add_argument("--beta", type=float, default=0.3, help="GT-band weight")
    ap.add_argument("--kd-T", type=float, default=2.0)
    ap.add_argument("--cldice", type=float, default=0.0,
                    help="soft-clDice weight (0=off) — sheet-connectivity loss; try 0.2")
    ap.add_argument("--cldice-iters", type=int, default=3,
                    help="soft-skeleton erosion rounds (~half the GT band half-thickness)")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient-accumulation micro-batches per optimizer step (effective batch = accum*batch)")
    ap.add_argument("--opt", choices=["adamw", "sgd"], default="adamw",
                    help="sgd = villa's recipe (lr 0.01, nesterov 0.99, wd 3e-5) as an ablation arm")
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="CE label smoothing (villa uses 0.1 in the self-distill pipeline)")
    ap.add_argument("--base", type=int, default=16, help="student base width")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out", default="student")
    ap.add_argument("--resume", default="")
    ap.add_argument("--qat", action="store_true", help="enable torchao int8 QAT (final phase)")
    ap.add_argument("--feed-timeout", type=float, default=300.0,
                    help="ring starvation timeout (s); cold big-patch starts legitimately take minutes")
    ap.add_argument("--val-ring", default="", help="held-out feed ring for periodic validation")
    ap.add_argument("--compile", action="store_true", help="torch.compile the student (max-autotune)")
    ap.add_argument("--fused-adam", action=argparse.BooleanOptionalAction, default=True,
                    help="fused AdamW kernel (measured free win; --no-fused-adam to disable)")
    ap.add_argument("--pinned", action="store_true", help="pinned-memory H2D staging for batches")
    ap.add_argument("--prof", action="store_true", help="per-phase step timing (data/fwd/bwd/opt)")
    ap.add_argument("--channels-last", action="store_true", help="channels_last_3d memory format")
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=4)
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    net = StudentUNet(base=args.base).to(dev)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last_3d)
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    # optimizer ablation arm: villa trains ALL published models with SGD lr=0.01 nesterov
    # (nnU-Net heritage); our default stays AdamW until the A/B says otherwise.
    if args.opt == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5, fused=args.fused_adam)
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
        # int8 fake-quant on activations (per-token asymmetric) + weights (per-channel
        # symmetric) — the standard int8 QAT recipe, applied to the conv/linear modules.
        from torchao.quantization import quantize_
        from torchao.quantization.qat import IntxFakeQuantizeConfig, QATConfig
        act_cfg = IntxFakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
        w_cfg = IntxFakeQuantizeConfig(torch.int8, "per_channel", is_symmetric=True)
        quantize_(net, QATConfig(activation_config=act_cfg, weight_config=w_cfg, step="prepare"))
        print("torchao int8 QAT enabled (fake-quant prepared)")

    if args.compile:
        net = torch.compile(net, mode="max-autotune")
        print("torch.compile enabled (first steps include autotune)")
    ring = FeedRing(args.ring)
    vring = FeedRing(args.val_ring) if args.val_ring else None
    pin_ct = pin_gt = pin_te = None
    if args.pinned:
        P = ring.patch
        pin_ct = torch.empty((args.batch, P, P, P), dtype=torch.uint8, pin_memory=True)
        pin_gt = torch.empty((args.batch, P, P, P), dtype=torch.uint8, pin_memory=True)
        if ring.channels == 3:
            pin_te = torch.empty((args.batch, P, P, P), dtype=torch.uint8, pin_memory=True)
    prof_t = {"data": 0.0, "h2d": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
    def _sync():
        if args.prof:
            torch.cuda.synchronize()
        return time.time()
    print(f"ring: {ring.nslots} slots, patch={ring.patch}, channels={ring.channels}")
    has_teacher = ring.channels == 3
    if not has_teacher and args.alpha > 0:
        print("no teacher channel in ring -> alpha forced to 0 (GT-only training)")
        args.alpha = 0.0

    t0, seen = time.time(), 0
    stats_path = f"{args.out}_stats.jsonl"
    stats_f = open(stats_path, "a")
    feed_wait = 0.0
    print(f"stats -> {stats_path}")
    for step in range(step0, args.steps):
        tw = time.time()
        b = ring.next_batch(args.batch, timeout_s=args.feed_timeout)
        feed_wait += time.time() - tw
        ct = torch.from_numpy(b["ct"]).to(dev, non_blocking=True)
        gt = torch.from_numpy(b["gt"]).to(dev, non_blocking=True)
        x = ct.unsqueeze(1).float()
        if args.channels_last:
            x = x.to(memory_format=torch.channels_last_3d)
        x = (x - x.mean(dim=(2, 3, 4), keepdim=True)) / (x.std(dim=(2, 3, 4), keepdim=True) + 1e-6)
        # tri-state GT (ml/rasterize.hpp): 255 sheet / 128 trusted background / 0 unlabeled.
        # Hard losses see ONLY labeled voxels — an unlabeled voxel may hold a sheet no segment
        # covers (multi-segment chunks), so punishing a positive there would be wrong.
        y = (gt == 255).float()
        known = (gt > 0).float()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
            ce = F.cross_entropy(logits.float(), y.long(), reduction="none", label_smoothing=args.label_smooth)
            ce = (ce * known).sum() / known.sum().clamp(min=1)
            dl = dice_loss(logits.float(), y, known)
            loss = args.beta * (dl + ce)
            cld = torch.zeros((), device=dev)
            if args.cldice > 0:
                cld = cldice_loss(logits.float(), y, known, args.cldice_iters)
                loss = loss + args.cldice * cld
            kd = torch.zeros((), device=dev)
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

        t2 = _sync()
        # gradient accumulation: N micro-batches per optimizer step (effective batch = N*batch).
        # Useless when the GPU is data-starved (same samples/step budget) — it's for effective-
        # batch scaling where VRAM caps the real batch (patch=256, the scaled L student).
        if step % args.accum == 0:
            opt.zero_grad(set_to_none=True)
        (loss / args.accum).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1e9).item()  # measure, don't clip
        t3 = _sync()
        prof_t["fwd_bwd"] += t3 - t2
        if (step + 1) % args.accum == 0:
            opt.step()
            sched.step()
        t4 = _sync()
        prof_t["opt"] += t4 - t3
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()):
                pe.lerp_(pn, 1 - args.ema)
            for be, bn in zip(ema.buffers(), net.buffers()):
                be.copy_(bn)

        seen += args.batch
        if args.prof and step % 50 == 0 and step > step0:
            n = 50.0
            print(f"prof: data {prof_t['data']/n*1e3:.0f}ms h2d {prof_t['h2d']/n*1e3:.0f}ms "
                  f"fwd+bwd {prof_t['fwd_bwd']/n*1e3:.0f}ms opt {prof_t['opt']/n*1e3:.0f}ms /step", flush=True)
            for k in prof_t: prof_t[k] = 0.0
        if step % 50 == 0:
            dt = time.time() - t0
            with torch.no_grad():
                # THE learning signal: mean predicted sheet-prob ON labeled sheet voxels vs on
                # trusted background. Separation growing = the model is learning; both stuck
                # near the positive-class prior = it isn't.
                prob = torch.softmax(logits.float(), 1)[:, 1]
                sheet_m = y > 0.5
                bg_m = (known > 0.5) & ~sheet_m
                p_sheet = prob[sheet_m].mean().item() if sheet_m.any() else float("nan")
                p_bg = prob[bg_m].mean().item() if bg_m.any() else float("nan")
                rec = {
                    "step": step,
                    "loss": round(loss.item(), 5),
                    "dice": round(dl.item(), 5),
                    "ce": round(ce.item(), 5),
                    "kd": round(kd.item(), 5),
                    "cld": round(cld.item(), 5),
                    "p_sheet": round(p_sheet, 4),   # want -> 1
                    "p_bg": round(p_bg, 4),         # want -> 0
                    "sep": round(p_sheet - p_bg, 4),
                    "gt_sheet_frac": round(sheet_m.float().mean().item(), 4),
                    "gt_known_frac": round(known.mean().item(), 4),
                    "grad_norm": round(gnorm, 3),
                    "lr": sched.get_last_lr()[0],
                    "patches_per_s": round(seen / max(dt, 1e-9), 2),
                    "feed_wait_frac": round(feed_wait / max(dt, 1e-9), 3),  # ~1 = feed-starved
                    "ring_ready": ring.ready_count(),
                    "gpu_gb": round(torch.cuda.memory_allocated() / 2**30, 2),
                }
                stats_f.write(json.dumps(rec) + "\n")
                stats_f.flush()
                print(f"step {step}  loss {rec['loss']:.4f} (dice {rec['dice']:.3f} ce {rec['ce']:.3f} "
                      f"kd {rec['kd']:.3f} cld {rec['cld']:.3f})  "
                      f"sep {rec['sep']:.3f} (sheet {rec['p_sheet']:.3f} bg {rec['p_bg']:.3f})  "
                      f"gnorm {rec['grad_norm']:.2f}  {rec['patches_per_s']:.1f} p/s  "
                      f"feedwait {rec['feed_wait_frac']:.0%}  ring {rec['ring_ready']}/{ring.nslots}",
                      flush=True)
        if vring is not None and step % args.val_every == 0 and step > step0:
            with torch.no_grad():
                net.eval()
                vloss = vsep = 0.0
                for _ in range(args.val_batches):
                    vb = vring.next_batch(args.batch, timeout_s=args.feed_timeout)
                    vx = torch.from_numpy(vb["ct"]).to(dev).unsqueeze(1).float()
                    vx = (vx - vx.mean(dim=(2, 3, 4), keepdim=True)) / (vx.std(dim=(2, 3, 4), keepdim=True) + 1e-6)
                    vgt = torch.from_numpy(vb["gt"]).to(dev)
                    vy = (vgt == 255).float()
                    vk = (vgt > 0).float()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vl = net(vx)
                    vce = F.cross_entropy(vl.float(), vy.long(), reduction="none")
                    vloss += ((vce * vk).sum() / vk.sum().clamp(min=1)).item() / args.val_batches
                    vp = torch.softmax(vl.float(), 1)[:, 1]
                    vs_m, vb_m = vy > 0.5, (vk > 0.5) & (vy <= 0.5)
                    if vs_m.any() and vb_m.any():
                        vsep += (vp[vs_m].mean() - vp[vb_m].mean()).item() / args.val_batches
                net.train()
                vrec = {"step": step, "val_ce": round(vloss, 5), "val_sep": round(vsep, 4)}
                stats_f.write(json.dumps(vrec) + "\n")
                stats_f.flush()
                print(f"step {step}  VAL ce {vloss:.4f}  sep {vsep:.3f}", flush=True)
        if step % args.ckpt_every == 0 and step > step0:
            path = f"{args.out}_step{step}.pt"
            torch.save({"net": net.state_dict(), "ema": ema.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step},
                       path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"checkpoint -> {path}", flush=True)

    torch.save({"ema": ema.state_dict(), "step": args.steps}, f"{args.out}_final.pt")
    # TorchScript export of the EMA — the artifact `fenix predict-surface` runs directly (.ts).
    try:
        ts = torch.jit.script(ema)
    except Exception:
        ex = torch.randn(1, 1, ring.patch, ring.patch, ring.patch, device=dev)
        ts = torch.jit.trace(ema, ex)
    ts.save(f"{args.out}_final.ts")
    print(f"done (exported {args.out}_final.ts)")


if __name__ == "__main__":
    main()
