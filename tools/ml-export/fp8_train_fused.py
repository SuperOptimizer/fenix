"""P2.2 — fused fp8-RESIDENT training block (conv + InstanceNorm + LeakyReLU) for sm120.

One autograd Function runs the whole CDNR block: fp8 conv (epilogue stats at N=1),
fused norm+act, fp8 re-quantized output. The backward reconstructs the pre-activation
and xhat FROM THE SAVED FP8 OUTPUT (LeakyReLU is invertible; the affine norm is
invertible per channel), so the block saves only: input fp8 (1B/act), output fp8
(1B/act — and it's the NEXT block's input), and per-(n,c) norm stats. No fp16
activation is ever saved → this is where both the 114 ms/step of layout-copy kernels
and the fp16-activation memory go away.

Conv bias is intentionally dropped in the fused forward: a per-channel constant is
absorbed exactly by the norm's mean subtraction, so the block output is bit-identical
and the bias's true gradient through this path is 0 (it simply gets no grad).

Usage: python fp8_train_fused.py     # per-block grad checks vs torch autograd (N=1, N=2,
                                     # stride-2, k=1, no-act) + micro-bench
"""
import os
import statistics
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from fp8_conv3d_op import (Fp8Tensor, fp8_conv3d_dgrad_s2, fp8_conv3d_v2,
                           fp8_conv3d_wgrad, in_norm_act, in_norm_act_bwd, in_stats,
                           pack_weight_dgrad_fp8, pack_weight_fp8, quantize_fp8)
from fp8_train import _amax_scale, swap_convs_fp8


class Fp8CDNR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w, gamma, beta, stride, neg_slope, packed):
        N, Ci, D, H, W = x.shape
        Co, _, k, _, _ = w.shape
        xm = x.permute(0, 2, 3, 4, 1)
        xm = (xm if xm.is_contiguous() else xm.contiguous()).reshape(-1, Ci)
        sx = _amax_scale(xm)
        x8 = quantize_fp8(xm, sx)
        if packed is None:
            w8, sw = pack_weight_fp8(w)
            wg8, swg = pack_weight_dgrad_fp8(w) if stride == 1 else (None, None)
        else:
            w8, sw, wg8, swg = packed
        xf = Fp8Tensor(x8, sx, (N, D, H, W, Ci))
        if N == 1:
            yv, mean, rstd = fp8_conv3d_v2(xf, w8, sw, Co, k, stride, xs_f=sx,
                                           want_stats=True)
            mean, rstd = mean.reshape(1, -1), rstd.reshape(1, -1)
        else:
            yv = fp8_conv3d_v2(xf, w8, sw, Co, k, stride, xs_f=sx)
            mean = rstd = None
        Do, Ho, Wo = yv.shape[2], yv.shape[3], yv.shape[4]
        hm = yv.permute(0, 2, 3, 4, 1).reshape(-1, Co)      # view back onto [M,C] base
        if mean is None:
            mean, rstd = in_stats(hm, N)
        pre16 = in_norm_act(hm, N, mean, rstd, gamma, beta, 1.0)
        # saved for backward: xhat DIRECTLY ((h-mean)*rstd — no 1/gamma amplification),
        # fp8 above 64k rows (noise averages out), fp16 below (deep stages, ~free);
        # plus the EXACT sign bits (branch by fp8-reconstructed sign flips ~0.4% of
        # near-zero elements). TODO: bitpack sgn + fold xhat emit into the norm kernel.
        M = hm.shape[0]
        xh32 = ((hm.reshape(N, -1, Co).float() - mean[:, None]) * rstd[:, None]
                ).reshape(M, Co)
        if M <= 65536:
            xh, sxh = xh32.half(), torch.ones(1, device=hm.device)
        else:
            sxh = _amax_scale(xh32)
            xh = quantize_fp8(xh32, sxh)
        sgn = (pre16 >= 0).to(torch.uint8)
        y16 = F.leaky_relu(pre16, neg_slope) if neg_slope != 1.0 else pre16
        ctx.save_for_backward(x8, sx, w8, sw, wg8, swg, xh, sgn, sxh, rstd, gamma)
        ctx.meta = (N, Ci, Co, D, H, W, Do, Ho, Wo, k, stride, neg_slope, x.dtype)
        return y16.reshape(N, Do, Ho, Wo, Co).permute(0, 4, 1, 2, 3)

    @staticmethod
    def backward(ctx, dout):
        x8, sx, w8, sw, wg8, swg, xh, sgn, sxh, rstd, gamma = ctx.saved_tensors
        N, Ci, Co, D, H, W, Do, Ho, Wo, k, stride, neg_slope, xdt = ctx.meta
        dm = dout.permute(0, 2, 3, 4, 1)
        dm = (dm if dm.is_contiguous() else dm.contiguous()).reshape(-1, Co)
        if dm.dtype != torch.float16:
            dm = dm.half()
        dh, dgamma, dbeta = in_norm_act_bwd(dm, xh, sgn, sxh, gamma, rstd, N, neg_slope)
        sdh = _amax_scale(dh)
        dh8 = quantize_fp8(dh, sdh)
        if stride == 1:
            dhf = Fp8Tensor(dh8, sdh, (N, D, H, W, Co))
            dx = fp8_conv3d_v2(dhf, wg8, swg, Ci, k, 1, out_dtype=torch.float16,
                               xs_f=sdh)
        else:
            dx = fp8_conv3d_dgrad_s2(dh8, w8, (N, D, H, W, Do, Ho, Wo), Ci, Co, k,
                                     sdh, sw)
        dw = fp8_conv3d_wgrad(dh8, x8, (N, D, H, W, Do, Ho, Wo), Ci, Co, k, stride,
                              sdh, sx)
        return (dx.to(xdt), dw, dgamma, dbeta, None, None, None)


class Fp8CDNRLayer(torch.nn.Module):
    """Drop-in for a ConvDropoutNormReLU (conv3d + InstanceNorm3d(affine) [+LeakyReLU]).
    Master params stay f32; packed fp8 weights cache on weight._version (as in
    Fp8Conv3dLayer); conv bias kept as a param but norm-absorbed (no grad)."""

    def __init__(self, mod):
        super().__init__()
        conv, norm = mod.conv, mod.norm
        nonlin = getattr(mod, "nonlin", None)
        self.weight = conv.weight
        self.bias = conv.bias
        self.gamma = norm.weight
        self.beta = norm.bias
        self.stride = conv.stride[0]
        self.neg_slope = (nonlin.negative_slope
                          if isinstance(nonlin, torch.nn.LeakyReLU) else 1.0)
        self._cache = (None, None)
        self.graph_mode = False

    def _packed(self):
        v = self.weight._version
        if self._cache[0] != v:
            wf = self.weight.detach().float()
            w8, sw = pack_weight_fp8(wf)
            wg8, swg = pack_weight_dgrad_fp8(wf) if self.stride == 1 else (None, None)
            self._cache = (v, (w8, sw, wg8, swg))
        return self._cache[1]

    def forward(self, x):
        return Fp8CDNR.apply(x, self.weight.float(), self.gamma.float(),
                             self.beta.float(), self.stride, self.neg_slope,
                             None if self.graph_mode else self._packed())


def swap_cdnr_fp8(net):
    """Replace every eligible ConvDropoutNormReLU with the fused block, THEN sweep the
    remaining bare convs/transpconvs with swap_convs_fp8. Returns (fused, convs, kept)."""
    from dynamic_network_architectures.building_blocks.simple_conv_blocks import \
        ConvDropoutNormReLU
    fused = kept = 0
    for name, mod in list(net.named_modules()):
        for cname, child in list(mod.named_children()):
            if not isinstance(child, ConvDropoutNormReLU):
                continue
            conv = child.conv
            norm = getattr(child, "norm", None)
            nonlin = getattr(child, "nonlin", None)
            drop = getattr(child, "dropout", None)
            k = conv.kernel_size
            ok = (isinstance(conv, torch.nn.Conv3d)
                  and isinstance(norm, torch.nn.InstanceNorm3d) and norm.affine
                  and abs(norm.eps - 1e-5) < 1e-12
                  and k[0] == k[1] == k[2] and k[0] in (1, 3) and conv.groups == 1
                  and conv.stride[0] == conv.stride[1] == conv.stride[2]
                  and conv.stride[0] in (1, 2)
                  and conv.padding == ((k[0] - 1) // 2,) * 3
                  and conv.dilation == (1, 1, 1)
                  and (drop is None or isinstance(drop, torch.nn.Identity)
                       or getattr(drop, "p", 0) == 0)
                  and (nonlin is None or isinstance(nonlin, (torch.nn.LeakyReLU,
                                                             torch.nn.Identity))))
            if ok:
                setattr(mod, cname, Fp8CDNRLayer(child))
                fused += 1
            else:
                kept += 1
    convs, kept2 = swap_convs_fp8(net)
    return fused, convs, kept + kept2


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def main():
    dev = "cuda"
    torch.manual_seed(0)
    print("== fused CDNR block grad checks vs torch autograd ==")
    for N, Ci, Co, P, k, stride, slope, tag in (
            (1, 64, 64, 32, 3, 1, 0.01, "N=1 k3 s1 lrelu"),
            (2, 64, 64, 32, 3, 1, 0.01, "N=2 k3 s1 lrelu"),
            (1, 64, 128, 32, 3, 2, 0.01, "N=1 k3 s2 lrelu"),
            (1, 64, 32, 32, 1, 1, 0.01, "N=1 k1 s1 lrelu"),
            (1, 64, 64, 32, 3, 1, 1.0, "N=1 k3 s1 NO-act")):
        conv = torch.nn.Conv3d(Ci, Co, k, stride=stride, padding=(k - 1) // 2,
                               bias=True).to(dev)
        norm = torch.nn.InstanceNorm3d(Co, affine=True).to(dev)
        with torch.no_grad():
            norm.weight.uniform_(0.5, 1.5)
            norm.bias.uniform_(-0.5, 0.5)
        x = (torch.randn(N, Ci, P, P, P, device=dev) * 0.5).requires_grad_(True)
        Po = P // stride
        gy = torch.randn(N, Co, Po, Po, Po, device=dev) * 0.1

        mod = types.SimpleNamespace(conv=conv, norm=norm,
                                    nonlin=torch.nn.LeakyReLU(slope) if slope != 1.0
                                    else None)
        lay = Fp8CDNRLayer(mod)
        y = lay(x)
        y.backward(gy)
        g8 = (x.grad.clone(), conv.weight.grad.clone(), norm.weight.grad.clone(),
              norm.bias.grad.clone())
        y8 = y.detach()
        x.grad = conv.weight.grad = norm.weight.grad = norm.bias.grad = None

        yr = norm(conv(x.float()))
        if slope != 1.0:
            yr = F.leaky_relu(yr, slope)
        yr.backward(gy.float())
        gr = (x.grad.clone(), conv.weight.grad.clone(), norm.weight.grad.clone(),
              norm.bias.grad.clone())
        x.grad = conv.weight.grad = norm.weight.grad = norm.bias.grad = None

        print(f"  {tag:>18}: y={corr(y8, yr):.4f} dx={corr(g8[0], gr[0]):.4f} "
              f"dw={corr(g8[1], gr[1]):.4f} dgamma={corr(g8[2], gr[2]):.4f} "
              f"dbeta={corr(g8[3], gr[3]):.4f}")

    print("\n== fused-block micro-bench vs torch (conv+IN+lrelu fwd+bwd, N=1 C=64 48^3) ==")
    Ci = Co = 64
    P = 48
    conv = torch.nn.Conv3d(Ci, Co, 3, padding=1, bias=True).to(dev)
    norm = torch.nn.InstanceNorm3d(Co, affine=True).to(dev)
    lay = Fp8CDNRLayer(types.SimpleNamespace(conv=conv, norm=norm,
                                             nonlin=torch.nn.LeakyReLU(0.01)))
    x0 = torch.randn(1, Ci, P, P, P, device=dev) * 0.5
    gy = torch.randn(1, Co, P, P, P, device=dev) * 0.1

    def f8():
        xr = x0.detach().requires_grad_(True)
        lay(xr).backward(gy)

    xh = x0.half()
    convh = torch.nn.Conv3d(Ci, Co, 3, padding=1).to(dev)
    normh = torch.nn.InstanceNorm3d(Co, affine=True).to(dev)

    def f16():
        xr = xh.detach().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.float16):
            y = F.leaky_relu(normh(convh(xr)), 0.01)
        y.backward(gy.half())

    def bench(fn, it=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(it):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(ts)

    t8, t16 = bench(f8), bench(f16)
    print(f"  fused-fp8 {t8:5.1f}ms | torch-fp16 {t16:5.1f}ms | {t16/t8:.2f}x")


if __name__ == "__main__":
    main()
