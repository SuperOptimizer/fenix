"""train — the FullScrollNet trainer: phantom pretrain -> real finetune -> QAT.

Phases run sequentially (`--phase phantom:20000 --phase real:30000 --phase
qat:4000`). Data comes from feeder.py: the phantom phase uses a pure-phantom
manifest (built on the fly if --manifest is not given); real/qat phases require
--manifest with regions. Precision: phantom/real = fp16 autocast + GradScaler
(bf16 = no scaler); qat = NO autocast, NO scaler (dy quantization IS the loss
scale — ml-export doctrine), custom int8/fp8 kernel lane via the swap functions.

QAT notes (review-verified): EMA freezes at the swap boundary (param names
change); validation/probe in the qat phase run the LIVE swapped net; the
accuracy gate is per-phantom surface_dice(net, pre-QAT reference) >= 0.998.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from feeder import FullScrollDataset, collate, make_loader
from losses import LossCfg, full_scroll_loss, head_metrics
from model import FullScrollNet, warm_start
from phantom import make_phantom

VAL_SEED0 = 1 << 31   # validation phantom seed space, disjoint from training


class Ema:
    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.module = copy.deepcopy(net).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        for pe, pn in zip(self.module.parameters(), net.parameters()):
            pe.lerp_(pn, 1 - self.decay)
        for be, bn in zip(self.module.buffers(), net.buffers()):
            be.copy_(bn)


def _pure_phantom_manifest(path, patch, seed):
    with open(path, "w") as f:
        f.write(f'[global]\npatch = {patch}\nphantom_frac = 1.0\n'
                f'seed = {seed}\nepoch_len = 1000000\n')
    return path


def held_out_phantoms(n, patch, bank=None):
    return [make_phantom(patch, seed=VAL_SEED0 + i, bank=bank) for i in range(n)]


class Trainer:
    def __init__(self, args):
        self.a = args
        if args.patch % 64 or args.patch < 128:
            raise SystemExit("--patch must be >=128 and divisible by 64 "
                             "(deepest of 7 stages needs >1 spatial element)")
        if args.warm and args.resume:
            raise SystemExit("--warm and --resume are mutually exclusive")
        torch.manual_seed(args.seed)
        self.dev = "cuda"
        self.net = FullScrollNet().to(self.dev)
        if args.warm:
            loaded, total = warm_start(self.net, args.warm)
            print(f"warm-start {loaded/1e6:.1f}M/{total/1e6:.1f}M params")
        self.ema = None if args.no_ema else Ema(self.net, args.ema_decay)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=args.lr,
                                     weight_decay=args.wd, fused=True)
        self.scaler = None if args.bf16 else torch.amp.GradScaler("cuda")
        self.amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        self.step = 0
        self.qat_swapped = False
        self.held = held_out_phantoms(args.val_phantoms, args.patch)
        self.plan = []
        for p in args.phase:
            name, n = p.split(":")
            if name not in ("phantom", "real", "qat"):
                raise SystemExit(f"unknown phase {name}")
            self.plan.append((name, int(n)))
        if any(n == "real" for n, _ in self.plan) and not args.manifest:
            raise SystemExit("--manifest required for a real phase")
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        self._stats = open(args.out + "_stats.jsonl", "a")
        self.best_val = math.inf
        if args.resume:
            self.load_ckpt(args.resume)

    # -- checkpointing -----------------------------------------------------
    def save_ckpt(self, path, extra=None):
        obj = {"net": self.net.state_dict(),
               "ema": self.ema.module.state_dict() if self.ema else None,
               "opt": self.opt.state_dict(),
               "scaler": self.scaler.state_dict() if self.scaler else None,
               "step": self.step, "plan": self.plan,
               "qat_swapped": self.qat_swapped, "args": vars(self.a),
               **(extra or {})}
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    def load_ckpt(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("qat_swapped"):
            self._swap_qat(calibrate=False)
        self.net.load_state_dict(ck["net"])
        if self.ema and ck.get("ema"):
            self.ema.module.load_state_dict(ck["ema"])
        self.opt.load_state_dict(ck["opt"])
        if self.scaler and ck.get("scaler"):
            self.scaler.load_state_dict(ck["scaler"])
        self.step = ck["step"]
        self.qat_swapped = ck.get("qat_swapped", False)
        print(f"resumed at step {self.step} (qat_swapped={self.qat_swapped})")

    def log(self, rec):
        rec["step"] = self.step
        rec["t"] = time.time()
        self._stats.write(json.dumps(rec) + "\n")
        self._stats.flush()

    # -- QAT machinery -----------------------------------------------------
    def _swap_qat(self, calibrate=True):
        sys.path.insert(0, os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "..", "ml-export"))
        from fp8_conv3d_op import load_tuned
        from fp8_train import (int8_calibrate, set_int8_qat, swap_convs_fp8,
                               swap_norms_fp8, swap_tails_fp8)
        tuned = os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json")
        if os.path.exists(tuned):
            load_tuned(tuned)
        self.ref_net = (self.ema.module if self.ema else
                        copy.deepcopy(self.net).eval())
        self.ref_net = copy.deepcopy(self.ref_net).eval()   # pre-QAT reference
        swap_convs_fp8(self.net)
        swap_norms_fp8(self.net)
        swap_tails_fp8(self.net)
        if self.a.qat == "int8":
            set_int8_qat(self.net, True)
            if calibrate:
                calib = [collate([s], "cuda")[0] for s in
                         (self._calib_samples())]
                int8_calibrate(self.net, calib)
        self.ema = None                    # param names changed; EMA frozen above
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=self.a.lr * 0.1,
                                     weight_decay=self.a.wd)
        self.qat_swapped = True

    def _calib_samples(self, n=8):
        ds = FullScrollDataset(self._manifest_path(), epoch=999)
        it = iter(ds)
        return [next(it) for _ in range(n)]

    def _manifest_path(self):
        if self.a.manifest:
            return self.a.manifest
        p = self.a.out + "_phantom_manifest.toml"
        if not os.path.exists(p):
            _pure_phantom_manifest(p, self.a.patch, self.a.seed)
        return p

    # -- validation / probe --------------------------------------------------
    @torch.no_grad()
    def validate(self):
        net = self.net if self.qat_swapped else \
            (self.ema.module if self.ema else self.net)
        net.eval()
        parts_acc, metrics_acc = {}, {}
        nb = 0
        from phantom import to_batch
        for i in range(0, min(len(self.held), self.a.val_batches * self.a.batch),
                       self.a.batch):
            x, t = to_batch(self.held[i:i + self.a.batch], self.dev)
            if self.qat_swapped:
                out = net(x)
            else:
                with torch.autocast("cuda", dtype=self.amp_dtype):
                    out = net(x)
            out = {k: v.float() for k, v in out.items()}
            _, parts = full_scroll_loss(out, t, self._loss_cfg(1.0))
            m = head_metrics(out, t)
            for k, v in parts.items():
                parts_acc[k] = parts_acc.get(k, 0) + float(v)
            for k, v in m.items():
                metrics_acc[k] = metrics_acc.get(k, 0) + v
            nb += 1
        self.net.train()
        rec = {("val_" + k): v / max(nb, 1) for k, v in
               {**parts_acc, **metrics_acc}.items()}
        if rec.get("val_w_mae", math.inf) < self.best_val and not self.qat_swapped:
            self.best_val = rec["val_w_mae"]
            self.save_ckpt(self.a.out + "_best.pt", {"val": rec})
        return rec

    @torch.no_grad()
    def probe(self):
        from wrap_probe import wrap_probe
        net = self.net if self.qat_swapped else \
            (self.ema.module if self.ema else self.net)
        net.eval()
        r = wrap_probe(lambda x: net(x), self.held[: self.a.probe_n],
                       device=self.dev, do_stitch=self.a.probe_stitch)
        self.net.train()
        return r

    def _loss_cfg(self, ramp):
        a = self.a
        return LossCfg(shell_dice_w=a.shell_dice, papyrus_dice_w=a.pap_dice,
                       phys_align_w=a.phys_align * ramp,
                       phys_pitch_w=a.phys_pitch * ramp,
                       ink_pos_weight=a.ink_pos_weight)

    # -- phases ---------------------------------------------------------------
    def run(self):
        for name, steps in self.plan:
            self.run_phase(name, steps)
        self.save_ckpt(self.a.out + "_final.pt")
        print("training complete:", self.a.out + "_final.pt")

    def run_phase(self, phase, steps):
        a = self.a
        print(f"=== phase {phase}: {steps} steps ===")
        if phase == "qat" and not self.qat_swapped:
            self._swap_qat()
        manifest = self._manifest_path() if phase != "phantom" or a.manifest \
            else _pure_phantom_manifest(a.out + "_phantom_manifest.toml",
                                        a.patch, a.seed)
        loader = make_loader(manifest, a.batch, a.workers, epoch=self.step)
        it = iter(loader)
        sma = deque(maxlen=a.sma)
        warmup = 200 if phase == "qat" else a.warmup
        base_lr = a.lr * (0.1 if phase == "qat" else 1.0)
        nan_streak = 0
        self.net.train()
        k0 = time.perf_counter()
        try:
            for k in range(steps):
                lr = base_lr * min(1.0, (k + 1) / max(warmup, 1)) * \
                    (0.5 * (1 + math.cos(math.pi * max(0, k - warmup)
                                         / max(steps - warmup, 1))))
                for g in self.opt.param_groups:
                    g["lr"] = lr
                tfeed = time.perf_counter()
                try:
                    x, t = next(it)
                except StopIteration:
                    it = iter(loader)
                    x, t = next(it)
                x = x.to(self.dev, non_blocking=True)
                t = {kk: v.to(self.dev, non_blocking=True) for kk, v in t.items()}
                feed_s = time.perf_counter() - tfeed
                cfg = self._loss_cfg(min(1.0, (k + 1) / max(a.phys_ramp, 1)))
                self.opt.zero_grad(set_to_none=True)
                if phase == "qat":
                    if self.a.qat == "int8" and (k + 1) % 250 == 0:
                        # frozen scales DRIFT with the weights (measured
                        # divergence at ~1500 steps) — refreeze periodically
                        from fp8_train import int8_calibrate
                        int8_calibrate(self.net, [x])
                    out = self.net(x)
                    loss, parts = full_scroll_loss(
                        {kk: v.float() for kk, v in out.items()}, t, cfg)
                    if not loss.isfinite():
                        nan_streak += 1
                        self.opt.zero_grad(set_to_none=True)
                        if nan_streak > 20:
                            self.save_ckpt(a.out + "_nan.pt")
                            raise SystemExit("20 consecutive non-finite losses")
                        continue
                    loss.backward()
                    self.opt.step()
                else:
                    with torch.autocast("cuda", dtype=self.amp_dtype):
                        out = self.net(x)
                    loss, parts = full_scroll_loss(
                        {kk: v.float() for kk, v in out.items()}, t, cfg)
                    if not loss.isfinite():
                        nan_streak += 1
                        self.opt.zero_grad(set_to_none=True)
                        if self.scaler:
                            self.scaler.update()
                        if nan_streak > 20:
                            self.save_ckpt(a.out + "_nan.pt")
                            raise SystemExit("20 consecutive non-finite losses")
                        continue
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.opt)
                        gn = torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), 1e9)
                        self.scaler.step(self.opt)
                        self.scaler.update()
                    else:
                        loss.backward()
                        gn = torch.nn.utils.clip_grad_norm_(
                            self.net.parameters(), 1e9)
                        self.opt.step()
                    if self.ema:
                        self.ema.update(self.net)
                nan_streak = 0
                self.step += 1
                sma.append(float(loss))
                if k % 50 == 0:
                    dt = (time.perf_counter() - k0) / max(k, 1)
                    self.log({"kind": "train", "phase": phase,
                              "loss_sma": float(np.mean(sma)), "lr": lr,
                              "feed_s": feed_s, "step_s": dt,
                              "mem_gib": torch.cuda.max_memory_allocated() / 2**30,
                              **{kk: float(v) for kk, v in parts.items()}})
                    print(f"  {phase} {k}/{steps} loss {np.mean(sma):.4f} "
                          f"({dt*1e3:.0f}ms/step)")
                if (k + 1) % a.val_every == 0:
                    rec = self.validate()
                    self.log({"kind": "val", "phase": phase, **rec})
                    print("  val:", {kk: round(v, 4) for kk, v in rec.items()
                                     if kk.startswith("val_w") or "dice" in kk})
                if (k + 1) % a.probe_every == 0:
                    rec = self.probe()
                    self.log({"kind": "probe", "phase": phase, **rec})
                    print("  probe:", rec)
                if (k + 1) % a.ckpt_every == 0:
                    self.save_ckpt(a.out + "_last.pt")
        finally:
            del it, loader
        if phase == "qat":
            self._qat_gate()
        self.save_ckpt(f"{a.out}_{phase}.pt")

    def _qat_gate(self):
        from fp8_forward import surface_dice
        from phantom import to_batch
        dices = []
        self.net.eval()
        with torch.no_grad():
            for ph in self.held[: self.a.val_phantoms]:
                x, _ = to_batch([ph], self.dev)
                p8 = self.net(x)["sem"][:, 0:1].float().sigmoid()[0, 0]
                with torch.autocast("cuda", dtype=torch.float16):
                    pr = self.ref_net(x)["sem"][:, 0:1].float().sigmoid()[0, 0]
                dices.append(float(surface_dice(p8, pr, tol_vox=2)))
        d = float(np.mean(dices))
        lvl = "OK" if d >= 0.998 else "WARN"
        print(f"QAT gate: surface_dice vs pre-QAT ref = {d:.4f} [{lvl}]")
        self.log({"kind": "qat_gate", "dice": d})
        from fp8_conv3d_op import dump_tuned
        dump_tuned(os.path.expanduser("~/.cache/fenix-fp8-tuned-train.json"))
        self.net.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", action="append", required=True,
                    help="name:steps; phantom|real|qat, repeatable, sequential")
    ap.add_argument("--manifest", default="", help="feeder manifest (real/qat)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--warm", default="")
    ap.add_argument("--resume", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--val-phantoms", type=int, default=8)
    ap.add_argument("--probe-every", type=int, default=1000)
    ap.add_argument("--probe-n", type=int, default=2)
    ap.add_argument("--probe-stitch", action="store_true")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--sma", type=int, default=20)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--qat", choices=["int8", "fp8"], default="int8")
    ap.add_argument("--shell-dice", type=float, default=0.5)
    ap.add_argument("--pap-dice", type=float, default=0.25)
    ap.add_argument("--ink-pos-weight", type=float, default=20.0)
    ap.add_argument("--phys-align", type=float, default=0.1)
    ap.add_argument("--phys-pitch", type=float, default=0.1)
    ap.add_argument("--phys-ramp", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    Trainer(ap.parse_args()).run()


if __name__ == "__main__":
    main()
