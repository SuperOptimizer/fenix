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
import contextlib
import copy
import json
import os
import sys
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
        # auxiliary material head (papyrus-vs-air): DENSE intensity-derived supervision that
        # shapes the encoder everywhere the sheet head is silent (ignore regions). Not part of
        # forward() so the exported .ts keeps the plain sheet-logits contract.
        self.aux_head = nn.Conv3d(c[0], 2, 1)

    def features(self, x):
        e0 = self.e0(x)
        e1 = self.e1(e0)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.d0(torch.cat([self.u0(d1), e0], 1))

    def forward(self, x, aux: bool = False):
        # aux=True returns (sheet_logits, aux_material_logits) — the training path.
        # Default stays the plain sheet-logits contract (.ts export, eval, DDP-safe:
        # everything routes through forward so DDP's reducer hooks always fire).
        f = self.features(x)
        if aux:
            return self.head(f), self.aux_head(f)
        return self.head(f)


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
    ap.add_argument("--wrapk", type=int, default=0,
                    help="instance mode: k wrap colors (ring classes 0 ignore/128 bg/150 sheet-unknown/200+c)")
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="CE label smoothing (villa uses 0.1 in the self-distill pipeline)")
    ap.add_argument("--aux-material", type=float, default=0.2,
                    help="aux papyrus-vs-air head weight (dense Otsu-derived supervision; 0=off)")
    ap.add_argument("--sma", type=int, default=20,
                    help="train-stats moving-average window in records (20 records = 1000 steps)")
    ap.add_argument("--warmup", type=int, default=1000,
                    help="linear LR warmup steps before the cosine (0=off)")
    ap.add_argument("--sma-val", type=int, default=5,
                    help="val moving-average window in val passes")
    ap.add_argument("--base", type=int, default=16, help="student base width")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--out", default="student")
    ap.add_argument("--resume", default="")
    ap.add_argument("--qat", action="store_true", help="enable torchao int8 QAT (final phase)")
    ap.add_argument("--fp8", action="store_true",
                    help="[BROKEN — DO NOT USE, kept for forensics] fp8 conv3d training lane. "
                         "The 2026-07-11 'loss parity' gates were fooled: CE fell by fitting "
                         "the background prior while SEPARATION stayed ~0 (A/B 2026-07-12: "
                         "bf16 sep 0.08->0.75 over 800 steps; fp8 flat ~0 at lr 3e-4 AND "
                         "1e-4, CE-only AND dice+CE; a live run sat at sep 0.07 for 11k "
                         "steps). Gate lesson: behavioral gates must track DISCRIMINATION "
                         "(sep/AUPRC) under the production loss stack, never loss alone.")
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
    ap.add_argument("--early-stop-vals", type=int, default=0,
                    help="stop after N consecutive vals without val-sep improvement (0 = off; "
                         "the best-val checkpoint is kept either way)")
    ap.add_argument("--val-batches", type=int, default=16)  # 4 was too noisy to read the plateau
    args = ap.parse_args()

    # --- DDP (torchrun --nproc-per-node N): rank r consumes ring "<ring>.r<r>" (the shm
    # ring is single-consumer — one feeder per rank), grads sync via DistributedDataParallel,
    # val/EMA/checkpoints/logs live on rank 0 only. Single-process runs are unchanged.
    ddp_rank = int(os.environ.get("RANK", "-1"))
    ddp = ddp_rank >= 0
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(os.environ.get("FENIX_DDP_BACKEND", "nccl"))
        local = int(os.environ.get("LOCAL_RANK", "0"))
        if os.environ.get("FENIX_DDP_BACKEND") != "gloo":
            torch.cuda.set_device(local)
        args.ring = f"{args.ring}.r{ddp_rank}"
        if ddp_rank != 0:
            args.val_ring = ""
    is_main = (not ddp) or ddp_rank == 0

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    net = StudentUNet(base=args.base, classes=(args.wrapk + 1) if args.wrapk > 0 else 2).to(dev)
    if args.channels_last:
        net = net.to(memory_format=torch.channels_last_3d)
    if args.fp8:
        assert not (args.qat or args.compile), "--fp8 is exclusive of --qat/--compile"
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml-export"))
        from fp8_train import swap_convs_fp8, swap_norms_fp8, swap_resblock_tails_fp8
        from fp8_conv3d_op import load_tuned
        tuned = os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json")
        if os.path.exists(tuned):
            print(f"fp8: pinned {load_tuned(tuned)} tuned configs (else first steps autotune ~min)")
        ns_, nk_ = swap_convs_fp8(net)
        print(f"fp8: {ns_} convs swapped (kept {nk_}), {swap_norms_fp8(net)} norms fused, "
              f"{swap_resblock_tails_fp8(net)} block tails fused")
    ema = copy.deepcopy(net).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    # optimizer ablation arm: villa trains ALL published models with SGD lr=0.01 nesterov
    # (nnU-Net heritage); our default stays AdamW until the A/B says otherwise.
    if args.opt == "sgd":
        opt = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.99, nesterov=True, weight_decay=3e-5)
    else:
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5, fused=args.fused_adam)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.steps - args.warmup))
    if args.warmup > 0:  # villa warms up 1500-2000 steps; our early gnorm spikes agree
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=args.warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[args.warmup])
    else:
        sched = cos
    step0 = 0
    best_vsep = -1.0
    plateau_vals = 0
    # moving averages: raw per-batch numbers are too noisy to narrate (measured: a 0.47-0.57
    # val band read as a fake 'phase transition'). SMA windows in STATS RECORDS (train records
    # every 50 steps -> sma=20 is a 1000-step window; val smoothed over sma_val passes).
    from collections import deque
    sma_w = {k: deque(maxlen=args.sma) for k in ("loss", "dice", "ce", "sep")}
    vsma_w = {k: deque(maxlen=args.sma_val) for k in ("val_ce", "val_sep")}
    def sma(d):
        vals = [v for v in d if v == v]  # drop NaN (bg-free batches)
        return sum(vals) / len(vals) if vals else float("nan")

    # fp8 norm fusion renames InstanceNorm affine params (weight/bias -> gamma/beta).
    # Checkpoints ALWAYS carry plain StudentUNet names so eval/deploy tooling never
    # sees fp8 internals; rename on the way out (save) and back in (resume).
    def plain_sd(m):
        # checkpoints ALWAYS carry plain StudentUNet names: strip torch.compile's
        # _orig_mod. wrapper prefix and (fp8-forensics mode) the fused-norm renames.
        sd = {}
        for k, v in m.state_dict().items():
            k = k.removeprefix("module.").removeprefix("_orig_mod.")
            sd[k] = v
        if not args.fp8:
            return sd
        return {k.replace(".gamma", ".weight").replace(".beta", ".bias"): v for k, v in sd.items()}

    def to_net_sd(sd, m):
        want = m.state_dict().keys()
        out = {}
        for k, v in sd.items():
            k = k.removeprefix("module.").removeprefix("_orig_mod.")
            if k not in want:
                k2 = k.replace(".weight", ".gamma").replace(".bias", ".beta")
                k = k2 if k2 in want else k
            out[k] = v
        return out

    if args.resume and os.path.exists(args.resume):
        st = torch.load(args.resume, map_location=dev, weights_only=False)
        net.load_state_dict(to_net_sd(st["net"], net))
        ema.load_state_dict(to_net_sd(st["ema"], ema))
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
    if ddp:
        from torch.nn.parallel import DistributedDataParallel
        dev_ids = None if os.environ.get("FENIX_DDP_BACKEND") == "gloo" else [int(os.environ.get("LOCAL_RANK", "0"))]
        net = DistributedDataParallel(net, device_ids=dev_ids)
        print(f"DDP rank {ddp_rank} up (ring {args.ring})")
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
    stats_path = f"{args.out}_stats.jsonl" if is_main else "/dev/null"
    stats_f = open(stats_path, "a")
    feed_wait = 0.0
    print(f"stats -> {stats_path}")

    # per-mesh loss telemetry: slot headers carry the sampled mesh id (feed.hpp writes the
    # id->path map to <ring>.meshes). A mesh whose patches stay high-CE late in training is a
    # label-noise suspect — clean labels fit, noise doesn't — so every run doubles as a label
    # audit. EMA (not lifetime mean) so early-training loss doesn't mask late misfits; bg draws
    # (mesh id 0xFFFFFFFF) are excluded — their labels come from every mesh on the volume.
    mesh_names = []
    if os.path.exists(args.ring + ".meshes"):
        mesh_names = [ln.strip() for ln in open(args.ring + ".meshes")]
    mesh_ce = {}   # id -> [ema, n]
    MESH_SENTINEL = 0xFFFFFFFF
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
        if args.wrapk > 0:
            # instance encoding: 0 ignore / 128 bg->cls0 / 150 sheet-unknown / 200+c -> cls 1+c
            cls = torch.full_like(gt.long(), -100)
            cls[gt == 128] = 0
            for c_i in range(args.wrapk):
                cls[gt == 200 + c_i] = 1 + c_i
            unk = (gt == 150).float()
            y = ((gt == 150) | (gt >= 200)).float()  # binary collapse for dice/sep reporting
            known = ((cls >= 0) | (gt == 150)).float()
        else:
            y = (gt == 255).float()
            known = (gt > 0).float()

        # auxiliary material labels (papyrus-vs-air), intensity-derived per sample: dense, mesh-
        # independent supervision. Per-sample Otsu valley on the RAW u8 ct; a +/-5% margin band
        # around the valley is ignored (-100 = F.cross_entropy default ignore_index).
        aux_y = None
        if args.aux_material > 0:
            aux_y = torch.full_like(gt.long(), -100)
            cf = ct.float()
            flat = cf.flatten(1)
            # Otsu per sample via 256-bin histogram (vectorized enough at batch 8)
            for i in range(cf.shape[0]):
                h = torch.histc(flat[i], bins=256, min=0, max=255)
                total = h.sum()
                idx = torch.arange(256, device=dev, dtype=torch.float32)
                w_b = torch.cumsum(h, 0)
                sum_b = torch.cumsum(h * idx, 0)
                w_f = total - w_b
                m_b = sum_b / w_b.clamp(min=1)
                m_f = (sum_b[-1] - sum_b) / w_f.clamp(min=1)
                var = w_b * w_f * (m_b - m_f) ** 2
                thr = var.argmax().float() + 0.5
                lo, hi = thr * 0.95, thr * 1.05
                air_frac = (flat[i] < thr).float().mean()
                if 0.03 < air_frac < 0.97:  # bimodality guard: skip mush patches entirely
                    aux_y[i][cf[i] < lo] = 0
                    aux_y[i][cf[i] > hi] = 1

        with (contextlib.nullcontext() if args.fp8 else torch.autocast("cuda", dtype=torch.bfloat16)):
            logits, aux_logits = net(x, True)
            if args.wrapk > 0:
                lf = logits.float()
                kc = (cls >= 0).float()
                kden = kc.sum(dim=(1, 2, 3)).clamp(min=1)
                # GAUGE-INVARIANT CE: absolute mod-k color is not locally inferable (run-1
                # verdict) — score against the BEST cyclic color shift per sample, so the
                # target is the RELATIVE coloring (order + contact distinctness).
                ce_shifts = []
                for s_i in range(args.wrapk):
                    cshift = torch.where(cls > 0, 1 + ((cls - 1 + s_i) % args.wrapk), cls)
                    cem = F.cross_entropy(lf, cshift.clamp(min=0), reduction="none")
                    ce_shifts.append((cem * kc).sum(dim=(1, 2, 3)) / kden)
                stacked = torch.stack(ce_shifts, 1)
                best_shift = stacked.argmin(dim=1)
                ce = stacked.min(dim=1).values.mean()
                cbest = torch.where(cls > 0, 1 + ((cls - 1 + best_shift.view(-1, 1, 1, 1)) % args.wrapk), cls)
                ce_map = F.cross_entropy(lf, cbest.clamp(min=0), reduction="none")
                # marginalized detection NLL on sheet-unknown voxels: -log sum_c p(sheet color c)
                logp = F.log_softmax(lf, dim=1)
                psheet = torch.logsumexp(logp[:, 1:], dim=1)
                det = -(psheet * unk).sum() / unk.sum().clamp(min=1)
                dl = det  # reported in the dice slot
                loss = args.beta * (ce + 2.0 * det)
            else:
                ce_map = F.cross_entropy(logits.float(), y.long(), reduction="none", label_smoothing=args.label_smooth)
                ce = (ce_map * known).sum() / known.sum().clamp(min=1)
                dl = dice_loss(logits.float(), y, known)
                loss = args.beta * (dl + ce)
            aux = torch.zeros((), device=dev)
            if aux_y is not None:
                aux = F.cross_entropy(aux_logits.float(), aux_y, ignore_index=-100)
                if aux.isfinite():
                    loss = loss + args.aux_material * aux
            # DDP anchor: aux participation is DATA-DEPENDENT (bimodality guard / all-ignore
            # batches) — a zero-weight touch marks aux_head used every iteration so the
            # reducer never sees unused parameters (cheaper than find_unused_parameters).
            loss = loss + 0.0 * aux_logits.float().sum()
            cld = torch.zeros((), device=dev)
            if args.cldice > 0:
                cld = cldice_loss(logits.float(), y, known, args.cldice_iters)
                loss = loss + args.cldice * cld
            kd = torch.zeros((), device=dev)
            if args.alpha > 0:
                tprob = (torch.from_numpy(b["teacher"]).to(dev).float() / 255.0).clamp(1e-4, 1 - 1e-4)
                T = args.kd_T
                # binary KL against the teacher's saturated prob, temperature-softened
                tsoft = ((tprob.logit()) / T).sigmoid()
                kd = F.binary_cross_entropy_with_logits(
                    (logits.float()[:, 1] - logits.float()[:, 0]) / T, tsoft) * (T * T)
                loss = loss + args.alpha * kd

        with torch.no_grad():
            ce_ps = ((ce_map.detach() * known).sum(dim=(1, 2, 3))
                     / known.sum(dim=(1, 2, 3)).clamp(min=1)).tolist()
        for i, m in enumerate(b["meta"]):
            if m[0] == MESH_SENTINEL or not (ce_ps[i] == ce_ps[i]):
                continue
            e = mesh_ce.setdefault(m[0], [ce_ps[i], 0])
            e[0] += 0.02 * (ce_ps[i] - e[0])
            e[1] += 1

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
                if not sheet_m.any() and args.alpha > 0 and "teacher" in b:
                    # mesh-free KD: no GT — report separation against the TEACHER's verdict
                    # (>0.5 sheet, <0.1 confident air) so the run still has a quality signal.
                    tp = torch.from_numpy(b["teacher"]).to(dev).float() / 255.0
                    sheet_m = tp > 0.5
                    bg_m = tp < 0.1
                p_sheet = prob[sheet_m].mean().item() if sheet_m.any() else float("nan")
                p_bg = prob[bg_m].mean().item() if bg_m.any() else float("nan")
                rec = {
                    "step": step,
                    "loss": round(loss.item(), 5),
                    "dice": round(dl.item(), 5),
                    "ce": round(ce.item(), 5),
                    "kd": round(kd.item(), 5),
                    "cld": round(cld.item(), 5),
                    "aux": round(aux.item(), 5),
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
                for kk in sma_w:
                    sma_w[kk].append(rec[kk])
                rec["sep_sma"] = round(sma(sma_w["sep"]), 4)
                rec["loss_sma"] = round(sma(sma_w["loss"]), 5)
                stats_f.write(json.dumps(rec) + "\n")
                stats_f.flush()
                print(f"step {step}  loss {rec['loss']:.4f} (dice {rec['dice']:.3f} ce {rec['ce']:.3f} "
                      f"kd {rec['kd']:.3f} cld {rec['cld']:.3f})  "
                      f"sep {rec['sep']:.3f}~{rec['sep_sma']:.3f} (sheet {rec['p_sheet']:.3f} bg {rec['p_bg']:.3f})  "
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
                    if args.wrapk > 0:
                        vcls = torch.full_like(vgt.long(), -100)
                        vcls[vgt == 128] = 0
                        for c_i in range(args.wrapk):
                            vcls[vgt == 200 + c_i] = 1 + c_i
                        vy = ((vgt == 150) | (vgt >= 200)).float()
                        vk = (vcls >= 0).float()
                    else:
                        vy = (vgt == 255).float()
                        vk = (vgt > 0).float()
                    with (contextlib.nullcontext() if args.fp8 else torch.autocast("cuda", dtype=torch.bfloat16)):
                        vl = net(vx)
                    if args.wrapk > 0:
                        vkden = vk.sum(dim=(1, 2, 3)).clamp(min=1)
                        vshift = []
                        for s_i in range(args.wrapk):
                            vcs = torch.where(vcls > 0, 1 + ((vcls - 1 + s_i) % args.wrapk), vcls)
                            vcem = F.cross_entropy(vl.float(), vcs.clamp(min=0), reduction="none")
                            vshift.append((vcem * vk).sum(dim=(1, 2, 3)) / vkden)
                        vloss += torch.stack(vshift, 1).min(dim=1).values.mean().item() / args.val_batches
                    else:
                        vce = F.cross_entropy(vl.float(), vy.long(), reduction="none")
                        vloss += ((vce * vk).sum() / vk.sum().clamp(min=1)).item() / args.val_batches
                    vp = (1.0 - torch.softmax(vl.float(), 1)[:, 0]) if args.wrapk > 0 else torch.softmax(vl.float(), 1)[:, 1]
                    vs_m, vb_m = vy > 0.5, (vk > 0.5) & (vy <= 0.5)
                    if vs_m.any() and vb_m.any():
                        vsep += (vp[vs_m].mean() - vp[vb_m].mean()).item() / args.val_batches
                net.train()
                vsma_w["val_ce"].append(vloss)
                vsma_w["val_sep"].append(vsep)
                vrec = {"step": step, "val_ce": round(vloss, 5), "val_sep": round(vsep, 4),
                        "val_ce_sma": round(sma(vsma_w["val_ce"]), 5),
                        "val_sep_sma": round(sma(vsma_w["val_sep"]), 4)}
                stats_f.write(json.dumps(vrec) + "\n")
                stats_f.flush()
                print(f"step {step}  VAL ce {vloss:.4f}~{vrec['val_ce_sma']:.4f}  "
                      f"sep {vsep:.3f}~{vrec['val_sep_sma']:.3f}", flush=True)
                # best-val checkpoint: the plateau is noisy, so the FINAL weights are rarely the
                # best weights — keep the val-sep winner for the eval gate / deployment.
                if vrec["val_sep_sma"] > best_vsep:  # select on the SMA, not one noisy pass
                    best_vsep = vrec["val_sep_sma"]
                    plateau_vals = 0
                    torch.save({"net": plain_sd(net), "ema": plain_sd(ema), "step": step,
                                "val_sep": vsep}, f"{args.out}_best.pt.tmp")
                    os.replace(f"{args.out}_best.pt.tmp", f"{args.out}_best.pt")
                    print(f"best-val checkpoint -> {args.out}_best.pt (sep {vsep:.3f})", flush=True)
                else:
                    plateau_vals += 1
                    if plateau_vals % 10 == 0:
                        print(f"plateau: no val-sep improvement in {plateau_vals} vals "
                              f"({plateau_vals * args.val_every} steps, best {best_vsep:.3f})", flush=True)
                    if args.early_stop_vals and plateau_vals >= args.early_stop_vals:
                        print(f"EARLY STOP: {plateau_vals} vals without improvement", flush=True)
                        break
        if is_main and step % args.ckpt_every == 0 and step > step0 and mesh_ce:
            rows = [{"mesh": mid,
                     "path": mesh_names[mid] if mid < len(mesh_names) else "",
                     "n": e[1], "ce_ema": round(e[0], 5)}
                    for mid, e in mesh_ce.items()]
            rows.sort(key=lambda r: -r["ce_ema"])
            with open(f"{args.out}_meshloss.json.tmp", "w") as mf:
                json.dump({"step": step, "meshes": rows}, mf, indent=1)
            os.replace(f"{args.out}_meshloss.json.tmp", f"{args.out}_meshloss.json")
            worst = ", ".join(f"{os.path.basename(r['path']) or r['mesh']}:{r['ce_ema']:.3f}"
                              for r in rows[:3])
            print(f"mesh-loss audit -> {args.out}_meshloss.json (worst: {worst})", flush=True)
        if is_main and step % args.ckpt_every == 0 and step > step0:
            path = f"{args.out}_step{step}.pt"
            torch.save({"net": plain_sd(net), "ema": plain_sd(ema),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step},
                       path + ".tmp")
            os.replace(path + ".tmp", path)
            print(f"checkpoint -> {path}", flush=True)

    if not is_main:
        return
    torch.save({"ema": plain_sd(ema), "step": args.steps}, f"{args.out}_final.pt")
    # TorchScript export of the EMA — the artifact `fenix predict-surface` runs directly (.ts).
    try:
        if args.fp8:
            plain = StudentUNet(base=args.base,
                                classes=(args.wrapk + 1) if args.wrapk > 0 else 2).to(dev).eval()
            plain.load_state_dict(plain_sd(ema))
            ts = torch.jit.script(plain)
        else:
            ts = torch.jit.script(ema)
    except Exception:
        ex = torch.randn(1, 1, ring.patch, ring.patch, ring.patch, device=dev)
        ts = torch.jit.trace(ema, ex)
    ts.save(f"{args.out}_final.ts")
    print(f"done (exported {args.out}_final.ts)")


if __name__ == "__main__":
    main()
