#!/usr/bin/env python3
"""finetune_ink_compression.py — compression-robustness self-distillation for ink_3d_dino_guided.

Finding (docs/design/training-data/findings/2026-07-19-q32-prediction-impact-ink-tta.md):
dct3d q=32 CT flips ink detections (pred-space max-abs 255 vs raw-CT input) while surface
survives. Fix: fine-tune the released ink net so its predictions on COMPRESSED CT match the
frozen teacher's predictions on RAW CT.

  teacher (frozen, released ckpt)  raw-CT patch      -> target prob (precomputed .npy)
  student (trainable, same init)   {raw | q32} patch -> BCE vs target, ink-weighted

Raw-input patches are mixed in (--raw-frac) so the model keeps its uncompressed behavior.
Normalization replicates the C++ inference contract exactly (per-patch 0.5/99.5 percentile
min-max, clamped to [0,1] — src/ml/infer.hpp norm_patch pct_minmax).

Data layout (per region NAME, produced by tools/train/prep_ink_ft_data.sh):
  <data>/NAME.raw.npy      u8 ZYX raw CT (ingest q=2, effectively lossless)
  <data>/NAME.q32.npy      u8 ZYX same CT transcoded at q=32
  <data>/NAME.teacher.npy  u8 ZYX teacher ink prob on raw CT (predict-scroll net=ink tta=8)

Usage:
  python3 finetune_ink_compression.py --ckpt <ckpt_78k_fullsup.pth> --data <dir> \
      [--val NAME] [--steps 6000] [--patch 128] [--batch 4] [--lr 2e-5] [--raw-frac 0.35] \
      [--out ink_ft_compression.pth]

Output ckpt keeps the upstream schema ({"ema_model": state_dict}) so
tools/ml-export/convert_weights.py --key ema_model converts it unchanged.
"""
import argparse
import glob
import os
import queue
import sys
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "ml-export"))
sys.path.insert(0, _HERE)
from reference import build_and_load  # noqa: E402


def pct_minmax(x):  # x [B,1,P,P,P] float in 0..255 — the C++ inference norm, per patch
    flat = x.flatten(1)
    lo = torch.quantile(flat, 0.005, dim=1, keepdim=True)
    hi = torch.quantile(flat, 0.995, dim=1, keepdim=True)
    return ((flat - lo) / (hi - lo).clamp_min(1e-6)).clamp_(0, 1).view_as(x)


class Regions:
    """Memmapped (raw, q32, teacher) triples + ink-biased patch sampling."""

    def __init__(self, data_dir, patch, exclude=(), pos_frac=0.7):
        self.patch, self.pos_frac = patch, pos_frac
        self.tri = []
        for rp in sorted(glob.glob(os.path.join(data_dir, "*.raw.npy"))):
            name = os.path.basename(rp)[: -len(".raw.npy")]
            if name in exclude:
                continue
            raw = np.load(rp, mmap_mode="r")
            q32 = np.load(rp.replace(".raw.npy", ".q32.npy"), mmap_mode="r")
            tch = np.load(rp.replace(".raw.npy", ".teacher.npy"), mmap_mode="r")
            assert raw.shape == q32.shape == tch.shape, name
            # Coarse ink-presence grid (16³ cells) for positive-biased origin draws.
            g = 16
            t = torch.from_numpy(np.ascontiguousarray(tch[:: g, :: g, :: g])).float()
            pos = (t > 25).nonzero()  # u8 threshold ≈ prob 0.1
            self.tri.append((name, raw, q32, tch, pos.numpy() * g))
        if not self.tri:
            raise SystemExit(f"no *.raw.npy under {data_dir} (after exclude={exclude})")
        print(f"regions: {[t[0] for t in self.tri]}", flush=True)

    def draw(self, rng):
        name, raw, q32, tch, pos = self.tri[rng.integers(len(self.tri))]
        P = self.patch
        dims = raw.shape
        if len(pos) and rng.random() < self.pos_frac:
            c = pos[rng.integers(len(pos))]
            org = [int(np.clip(c[i] - P // 2 + rng.integers(-P // 4, P // 4 + 1), 0, dims[i] - P)) for i in range(3)]
        else:
            org = [int(rng.integers(0, dims[i] - P + 1)) for i in range(3)]
        z, y, x = org
        sl = np.s_[z : z + P, y : y + P, x : x + P]
        return raw[sl], q32[sl], tch[sl]


def batch_tensors(regions, rng, batch, raw_frac, dev=None):
    # PAIRED batches: batch//2 locations, each contributing its raw AND its q32 patch
    # (raw half first, q32 half second) so the loss can demand raw↔q32 consistency at
    # identical coordinates. raw_frac is unused (kept for CLI compat).
    # CPU-only on purpose: under the Thunder Compute shim (network-attached GPU), CUDA
    # calls from a second thread wedge the process — every GPU op stays on the main
    # thread (2026-07-20 stall root-cause).
    xr, xq, ts = [], [], []
    for _ in range(batch // 2):
        raw, q32, tch = regions.draw(rng)
        xr.append(torch.from_numpy(np.ascontiguousarray(raw)).float())
        xq.append(torch.from_numpy(np.ascontiguousarray(q32)).float())
        ts.append(torch.from_numpy(np.ascontiguousarray(tch)).float() / 255.0)
    x = torch.stack(xr + xq).unsqueeze(1)
    t = torch.stack(ts + ts).unsqueeze(1)
    return x, t


class Prefetcher:
    """CPU batch assembly (memmap reads + stack) overlapped with the GPU step.
    Depth 3 keeps the GPU fed even when a draw crosses cold memmap pages on few vCPUs."""

    def __init__(self, regions, rng, batch, raw_frac, dev, depth=3):
        self.q = queue.Queue(maxsize=depth)
        self.stop = False

        def work():
            try:
                while not self.stop:
                    self.q.put(batch_tensors(regions, rng, batch, raw_frac, dev))
            except BaseException as e:  # surface worker death instead of hanging the loop
                self.q.put(e)

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def next(self):
        item = self.q.get()
        if isinstance(item, BaseException):
            raise item
        return item


def ink_weighted_bce(logits, target, pos_w=2.0):
    # Soft-target BCE vs the teacher's prob field. Mild ink emphasis only: dense soft
    # targets don't collapse to all-zero (that failure mode belongs to hard sparse GT),
    # and heavy pos_w measurably mis-calibrates the student away from the teacher
    # (2026-07-20 run 1: val MAE-vs-teacher rose monotonically at pos_w=10).
    w = 1.0 + (pos_w - 1.0) * (target > 0.1).float()
    return (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * w).sum() / w.sum()


def paired_loss(logits, target, pos_w, cons_w=1.0, anchor_w=2.0):
    # logits/target stacked [raw half | q32 half] at identical locations. BCE ties both
    # halves to the teacher; the consistency term is the direct objective — the q32
    # prediction must match the model's OWN raw prediction (detached: q32 chases raw,
    # never the reverse). anchor_w upweights the RAW half's BCE: without it the model
    # satisfies consistency by meeting in a dimmer middle (2026-07-20 run 2 referee:
    # blob recall recovered 70%→95% on q32 but confidence globally deflated — the raw
    # side must stay pinned to the teacher so q32 does all the moving).
    lr_, lq_ = logits.chunk(2)
    tr_, tq_ = target.chunk(2)
    bce = (anchor_w * ink_weighted_bce(lr_, tr_, pos_w) + ink_weighted_bce(lq_, tq_, pos_w)) / (anchor_w + 1.0)
    pr, pq = torch.sigmoid(lr_), torch.sigmoid(lq_)
    cons = (pq - pr.detach()).abs().mean()
    return bce + cons_w * cons, bce, cons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", default="", help="region NAME held out for validation")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--raw-frac", type=float, default=0.35)
    ap.add_argument("--pos-w", type=float, default=2.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--out", default="ink_ft_compression.pth")
    ap.add_argument("--seed", type=int, default=27)
    ap.add_argument("--resume", default="", help="checkpoint from a prior run of THIS script: "
                    "restores model+ema and continues at its saved step (Thunder wedge recovery)")
    ap.add_argument("--student-base", type=int, default=0,
                    help=">0: KD into a from-scratch StudentUNet(base, classes=1) instead of "
                         "fine-tuning the full net (base=64 ≈ 23M params ≈ 12x smaller); "
                         "raise --lr (~3e-4) and --steps for from-scratch training")
    ap.add_argument("--student-stem", type=int, default=2,
                    help="student stem stride (2 = half-res net + upsampled output, ~2.6x faster)")
    args = ap.parse_args()

    dev = "cuda"
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.student_base > 0:
        from train import StudentUNet  # tools/train/train.py — the surface-KD student arch
        import copy

        student = StudentUNet(base=args.student_base, classes=1, stem_stride=args.student_stem).to(dev).train()
        ema = copy.deepcopy(student).eval()
        n = sum(p.numel() for p in student.parameters())
        print(f"student: StudentUNet(base={args.student_base}, stem={args.student_stem}) "
              f"{n / 1e6:.1f}M params", flush=True)
    else:
        student = build_and_load(args.ckpt, ink=True).to(dev).train()
        ema = build_and_load(args.ckpt, ink=True).to(dev).eval()
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        student.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema_model"])
        start_step = int(ck.get("step", 0))
        print(f"resumed {args.resume} at step {start_step}", flush=True)
    for p in ema.parameters():
        p.requires_grad_(False)

    exclude = (args.val,) if args.val else ()
    train_regions = Regions(args.data, args.patch, exclude=exclude)
    val_regions = Regions(args.data, args.patch, exclude=tuple(t[0] for t in train_regions.tri)) if args.val else None

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    scaler = torch.amp.GradScaler("cuda")

    # VRAM probe (fill-the-card rule): if the requested batch OOMs, halve until a
    # forward+backward fits; if it fits easily, the operator should raise --batch.
    while True:
        try:
            x, t = batch_tensors(train_regions, rng, args.batch, args.raw_frac)
            x, t = pct_minmax(x.to(dev)), t.to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                probe, _, _ = paired_loss(student(x), t, args.pos_w)
            scaler.scale(probe).backward()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            print(f"batch={args.batch} fits: peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
                  f"of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}", flush=True)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            args.batch = max(1, args.batch // 2)
            print(f"OOM — retrying with batch={args.batch}", flush=True)

    for _ in range(start_step):  # burn the schedule forward on resume
        sched.step()
    pf = Prefetcher(train_regions, rng, args.batch, args.raw_frac, dev)
    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        x, t = pf.next()
        x, t = pct_minmax(x.to(dev)), t.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            loss, bce, cons = paired_loss(student(x), t, args.pos_w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        with torch.no_grad():
            for pe, ps in zip(ema.parameters(), student.parameters()):
                pe.lerp_(ps, 1.0 - args.ema)
            for be, bs in zip(ema.buffers(), student.buffers()):
                be.copy_(bs)

        # First steps print eagerly: the supervising watchdog kills on log silence, and a
        # cold start can legitimately take minutes to reach step 50 (2026-07-20: the stall
        # detector repeatedly killed healthy student warmups).
        if step <= 5 or step % 50 == 0:
            print(f"step {step}/{args.steps} loss {loss.item():.5f} bce {bce.item():.5f} "
                  f"cons {cons.item():.5f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{(time.time() - t0) / (step - start_step):.2f}s/step", flush=True)
        if val_regions and step % args.val_every == 0:
            student.eval()
            with torch.no_grad():
                maes_q, maes_r = [], []
                vrng = np.random.default_rng(1)
                for _ in range(16):
                    raw, q32, tch = val_regions.draw(vrng)
                    tt = torch.from_numpy(np.ascontiguousarray(tch)).float().to(dev) / 255.0
                    for src, acc in ((q32, maes_q), (raw, maes_r)):
                        xv = pct_minmax(torch.from_numpy(np.ascontiguousarray(src)).float()[None, None].to(dev))
                        with torch.autocast("cuda", dtype=torch.float16):
                            pv = torch.sigmoid(ema(xv))[0, 0].float()
                        acc.append((pv - tt).abs().mean().item() * 255)
                print(f"  VAL step {step}: EMA MAE-vs-teacher q32 {np.mean(maes_q):.3f} raw {np.mean(maes_r):.3f} "
                      f"spread {np.mean(maes_q) - np.mean(maes_r):+.3f} (u8, ink-biased patches)", flush=True)
            student.train()
            torch.save({"ema_model": ema.state_dict(), "model": student.state_dict(), "step": step}, args.out)

    torch.save({"ema_model": ema.state_dict(), "model": student.state_dict(), "step": args.steps}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
