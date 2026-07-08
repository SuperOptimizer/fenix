"""lanes — uniform inference interface over FullScrollNet precision lanes.

InferFn contract: __call__(x f32 [b,1,P,P,P] z-scored, origins list) ->
f32/f16 [b,10,P,P,P] in CHANNEL_ORDER with sem/interior/ink already sigmoided.
Origins are global patch origins (used only by the fake lane).

Lanes (review-reconciled): fp16cl (default), fp8 / int8 via the module-SWAP
path (fp8_train) — NEVER the fp8_forward monkeypatch path on a swapped net
(mutually exclusive mechanisms); int8 statically calibrated with tensor
patches; trt raises NotImplementedError until an engine builder exists; fake =
CPU analytic spiral for e2e tests. No TTA (single-axis flips reverse winding
chirality — no per-channel fixup exists for the wind head).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "ml-export"))

CHANNEL_ORDER = ["papyrus", "recto", "verso", "interior",
                 "normal_z", "normal_y", "normal_x", "w_sin", "w_cos", "ink"]
NCH = len(CHANNEL_ORDER)


class HeadCat(torch.nn.Module):
    """FullScrollNet -> [b,10,...] in CHANNEL_ORDER; sigmoids applied."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        o = self.net(x)
        return torch.cat([o["sem"].sigmoid(), o["interior"].sigmoid(),
                          o["normal"], o["wind"], o["ink"].sigmoid()], dim=1)


def _load_net(ckpt, device):
    from model import FullScrollNet
    net = FullScrollNet().eval()
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("ema") or ck.get("net") or ck
    net.load_state_dict(sd)
    return net.to(device)


class TorchLane:
    def __init__(self, net, batch, patch, device, cl=True):
        self.fn = HeadCat(net)
        self.batch, self.patch, self.device = batch, patch, device
        self.cl = cl

    @torch.inference_mode()
    def __call__(self, x, origins=None):
        if self.cl:
            x = x.to(memory_format=torch.channels_last_3d)
        return self.fn(x)


def load_lane(lane, ckpt, device="cuda", patch=128, batch=2,
              calib_patches=None,
              tuned_json="~/.cache/fenix-fp8-tuned-train.json"):
    if lane == "fake":
        return FakeLane(patch)
    if lane == "trt":
        raise NotImplementedError(
            "build the engine first: tools/ml-export export path for the "
            "HeadCat wrapper, then wire a TrtNet-style twin here")
    net = _load_net(ckpt, device)
    if lane == "fp16cl":
        net = net.half().to(memory_format=torch.channels_last_3d)
        base = TorchLane(net, batch, patch, device)
        fn = base.fn

        @torch.inference_mode()
        def call(x, origins=None):
            return fn(x.half().to(memory_format=torch.channels_last_3d)).float()
        call_obj = type("Lane", (), {"batch": batch, "patch": patch,
                                     "__call__": staticmethod(call)})()
        return call_obj
    if lane in ("fp8", "int8"):
        from fp8_conv3d_op import load_tuned
        from fp8_train import (int8_calibrate, set_int8_qat, swap_convs_fp8,
                               swap_norms_fp8, swap_tails_fp8)
        tj = os.path.expanduser(tuned_json)
        if os.path.exists(tj):
            load_tuned(tj)
        swap_convs_fp8(net)
        swap_norms_fp8(net)
        swap_tails_fp8(net)
        if lane == "int8":
            set_int8_qat(net, True)
            if not calib_patches:
                raise RuntimeError("int8 lane needs calib_patches (tensors)")
            int8_calibrate(net, [p.to(device) for p in calib_patches])
        net.eval()
        return TorchLane(net, batch, patch, device, cl=False)
    raise KeyError(lane)


class FakeLane:
    """CPU analytic spiral over GLOBAL coords — powers the e2e phantom test.
    Parameters chosen to match phantom defaults (pitch 14, thickness 5)."""
    batch = 4

    def __init__(self, patch, pitch=14.0, thickness=5.0, center=(512.0, 512.0)):
        self.patch = patch
        self.pitch, self.thickness = pitch, thickness
        self.center = center

    def __call__(self, x, origins=None):
        b = x.shape[0]
        P = self.patch
        out = torch.zeros(b, NCH, P, P, P, dtype=torch.float32)
        for i in range(b):
            oz, oy, ox = origins[i] if origins else (0, 0, 0)
            z, y, xx = np.meshgrid(np.arange(oz, oz + P, dtype=np.float64),
                                   np.arange(oy, oy + P, dtype=np.float64),
                                   np.arange(ox, ox + P, dtype=np.float64),
                                   indexing="ij")
            ry, rx = y - self.center[0], xx - self.center[1]
            r = np.hypot(ry, rx) + 1e-6
            W = r / self.pitch - np.arctan2(ry, rx) / (2 * np.pi)
            fd = W - np.round(W)
            th = self.thickness / (2 * self.pitch)
            pap = (np.abs(fd) < th).astype(np.float32)
            gz, gy, gx = np.gradient(W)
            for g in (gz, gy, gx):
                g[np.abs(g) > 0.25] = 0.0
            nn = np.sqrt(gz**2 + gy**2 + gx**2) + 1e-9
            interior = (r > self.pitch * 0.75).astype(np.float32)
            ang = 2 * np.pi * (W % 1.0)
            ch = [pap,
                  pap * (fd > th - 0.35 * th),
                  pap * (fd < -th + 0.35 * th),
                  interior,
                  gz / nn, gy / nn, gx / nn,
                  np.sin(ang), np.cos(ang),
                  np.zeros_like(pap)]
            out[i] = torch.from_numpy(np.stack(ch).astype(np.float32))
        return out
