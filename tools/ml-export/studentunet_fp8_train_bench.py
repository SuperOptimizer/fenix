#!/usr/bin/env python3
"""StudentUNet fp8/int8 TRAINING bench: does the sm120 recipe transfer off the teacher?

Inference verdict is in (TRT fp16 29.5ms, QDQ dead, resident port shelved) — training
is where wall-clock hurts (studentM: ~20h/60k steps). The training swap machinery
(swap_convs_fp8 & friends) walks named_modules, so unlike the class-bound resident
inference path it should bind StudentUNet directly. This measures it:

  lane fp16 : autocast fp16 + cuDNN (the production train.py path)
  lane fp8  : swap_convs_fp8 (+norm/tail fuse where they bind — StudentUNet's
              functional F.leaky_relu means they may no-op) — full fp8 fwd+bwd
  lane i8f8 : int8 forward (delayed scaling) + fp8 backward — the frozen sm120 recipe

Identical perturbed init, identical real-CT batches, CE loss vs the codec-decoded band
GT (sheet>=192 / bg 64..191 / ignore<64). Reports ms/step (median), loss curves, and
step-1 grad correlation (median over params vs the fp16 twin).

Usage: studentunet_fp8_train_bench.py [--steps 30] [--batch 4] [--base 32]
"""
import sys, os, time, statistics, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
import numpy as np
import torch
import torch.nn.functional as F

from train import StudentUNet
from fp8_conv3d_op import load_tuned, dump_tuned


def band_gt(gt):
    sheet = gt >= 192
    lab = gt >= 64
    y = np.zeros(gt.shape, dtype=np.int64)
    y[sheet] = 1
    return y, lab


def batches(crops, batch, patch, steps, seed=3):
    rng = np.random.default_rng(seed)
    import glob
    cds = sorted(glob.glob(f"{crops}/crop*"))
    cts = [np.load(f"{c}/ct.npy") for c in cds]
    gts = [np.load(f"{c}/gt.npy") for c in cds]
    for _ in range(steps):
        xs, ys, ms = [], [], []
        for _ in range(batch):
            i = rng.integers(len(cts))
            D = cts[i].shape[0]
            o = rng.integers(0, D - patch + 1, 3)
            sl = tuple(slice(a, a + patch) for a in o)
            xs.append(cts[i][sl].astype(np.float32) / 255.0)
            y, m = band_gt(gts[i][sl])
            ys.append(y)
            ms.append(m)
        yield (torch.from_numpy(np.stack(xs))[:, None].cuda(),
               torch.from_numpy(np.stack(ys)).cuda(),
               torch.from_numpy(np.stack(ms)).cuda())


def loss_fn(logits, y, m):
    ce = F.cross_entropy(logits, y, reduction="none")
    return (ce * m).sum() / m.sum().clamp(min=1)


def run_lane(tag, net, data, lr, autocast_fp16):
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    ts, losses = [], []
    g1 = None
    for i, (x, y, m) in enumerate(data):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        if autocast_fp16:
            with torch.autocast("cuda", torch.float16):
                lo = net(x)
                loss = loss_fn(lo.float(), y, m)
        else:
            lo = net(x)
            loss = loss_fn(lo.float(), y, m)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if i == 0:
            g1 = {n: p.grad.detach().float().clone() for n, p in net.named_parameters()
                  if p.grad is not None}
        opt.step()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        losses.append(loss.item())
    med = statistics.median(ts[3:])
    print(f"{tag}: {med:.0f} ms/step | loss {losses[0]:.4f} -> {losses[-1]:.4f} "
          f"(last5 {np.mean(losses[-5:]):.4f})", flush=True)
    return med, losses, g1


def grad_corr(ga, gb):
    cs = []
    for n in ga:
        a, b = ga[n].flatten(), gb.get(n, None)
        if b is None or a.numel() < 32:
            continue
        b = b.flatten()
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        cs.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    return float(np.median(cs)), len(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--crops", default="/tmp/gtqc/m8/eval")
    args = ap.parse_args()
    torch.manual_seed(0)
    TUNED = os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json")
    if os.path.exists(TUNED):
        print(f"pinned {load_tuned(TUNED)} tuned configs")

    def fresh():
        torch.manual_seed(0)
        n = StudentUNet(base=args.base).cuda()
        g = torch.Generator(device="cuda").manual_seed(7)
        with torch.no_grad():
            for p in n.parameters():
                if p.dim() >= 2:
                    p.add_(torch.randn(p.shape, generator=g, device="cuda")
                           * (0.03 * p.std().clamp(min=1e-8)))
        return n.train()

    def data():
        return batches(args.crops, args.batch, args.patch, args.steps)

    _, _, gref = None, None, None
    t16, l16, gref = run_lane("fp16 ", fresh(), data(), args.lr, True)

    from fp8_train import swap_convs_fp8, swap_norms_fp8, swap_tails_fp8
    net8 = fresh()
    ns, nk = swap_convs_fp8(net8)
    try:
        nn_ = swap_norms_fp8(net8)
    except Exception as e:  # noqa: BLE001 — functional acts may defeat fusion
        nn_ = f"failed:{type(e).__name__}"
    try:
        nt = swap_tails_fp8(net8)
    except Exception as e:  # noqa: BLE001
        nt = f"failed:{type(e).__name__}"
    print(f"fp8 swap: {ns} convs (kept {nk}), norms {nn_}, tails {nt}", flush=True)
    t8, l8, g8 = run_lane("fp8  ", net8, data(), args.lr, False)
    c8, n8 = grad_corr(gref, g8)

    from fp8_train import set_int8_qat, set_int8_bwd_fp8
    neti = fresh()
    swap_convs_fp8(neti)
    try:
        swap_norms_fp8(neti)
        swap_tails_fp8(neti)
    except Exception:  # noqa: BLE001
        pass
    set_int8_qat(neti)
    set_int8_bwd_fp8(neti)
    ti, li, gi = run_lane("i8f8 ", neti, data(), args.lr, False)
    ci, ni = grad_corr(gref, gi)

    print(f"pinned {dump_tuned(TUNED)} tuned configs -> {TUNED}")
    print(f"\nVERDICT: fp16 {t16:.0f}ms | fp8 {t8:.0f}ms ({t16/t8:.2f}x, grad-corr "
          f"{c8:.4f}/{n8}p) | i8f8 {ti:.0f}ms ({t16/ti:.2f}x, grad-corr {ci:.4f}/{ni}p)")
    print("gate: adopt if speedup >=1.15x AND grad-corr >=0.93 AND loss curves track")


if __name__ == "__main__":
    main()
