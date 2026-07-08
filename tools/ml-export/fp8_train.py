"""fp8 conv3d TRAINING: autograd Function + grad-correctness + step harness (sm120).

Forward = the cube-tiled v2 implicit-GEMM kernel (fp8 x, fp8 W -> f32 acc -> fp16 out).
Backward:
  dgrad — for stride-1 same-pad convs, dx = conv3d(dy, W') where W'[ci,tap,co] =
          W[co, mirror(tap), ci]: the SAME v2 forward kernel with a repacked weight
          (pack_weight_dgrad_fp8). No new kernel.
  wgrad — dW[co,tap,ci] = sum_m dy[m,co] x[shift(m,tap),ci]: per-tap GEMM with the
          M-reduction split across CTAs + f32 atomic accumulation (_wgrad_f8).
Both backward operands are fp8 e4m3 in this spike (production would use e5m2 for dy —
wider range; same kernels, dtype swap). Accumulation is f32 everywhere. The saved-for-
backward activation is the fp8 tensor itself: 1 byte/act — a 2x activation-memory saving
vs fp16 training on top of the speed.

ConvTranspose3d k2s2 (the 6 decoder upsamplers) is fp8 too: fwd = ONE fp8 GEMM
[M,Cin]@[Cin,8*Cout] + pixel-shuffle (same trick as inference), bwd = two more fp8
GEMMs (dx = dy_g @ W^T, dW = x^T @ dy_g) via torch._scaled_mm — no new Triton kernel.

Usage: python fp8_train.py            # grad-correctness vs fp32 autograd + micro-bench
       python fp8_train.py --steps 30 # + optimizer-loop sanity (loss must fall)
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from fp8_conv3d_op import (E4M3_MAX, XHAT_SCALE, Fp8Tensor, quantize_i8_delayed,
                           quantize_i8_dyn, fp8_conv3d_dgrad_s2,
                           fp8_conv3d_v2, fp8_conv3d_wgrad, in_norm_act_bwd,
                           in_norm_act_train, in_stats, pack_weight_dgrad_fp8,
                           pack_weight_fp8, pack_weight_dgrad_i8, pack_weight_i8, quantize_fp8,
                           quantize_i8_affine, quantize_i8_fused, quantize_i8_static, tail_train_bwd,
                           tail_train_fwd, wsum_i8)


def _q(t_mc, scale):
    return (t_mc / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


def _amax_scale(t_mc):
    """Per-tensor fp8 scale via inf-norm — no abs() materialization, f32 math (fp16
    scale math underflows -> NaN; see the trap note above)."""
    return (torch.linalg.vector_norm(t_mc, torch.inf).float().clamp(min=1e-12)
            / E4M3_MAX).reshape(1)


class Fp8Conv3d(torch.autograd.Function):
    """same-pad conv3d (k in {1,3}, stride in {1,2}) + optional bias:
    fp8 fwd + fp8 bwd (dgrad+wgrad), f32 accumulate. `packed` (optional) is the
    quantize-on-update cache: (w8, swf, wg8, swgf) with scales as python floats —
    None means pack inline (standalone use)."""

    @staticmethod
    def forward(ctx, x, w, bias, stride, packed=None, stats_out=None, i8packed=None,
                i8cal=None, cal_out=None, bsmask=None, dyn_state=None,
                bwd_fp8=False):
        # x [N,Ci,D,H,W] fp16/f32; w f32 [Co,Ci,k,k,k]. stats_out: mutable list — at
        # N=1 the conv epilogue computes the following InstanceNorm's (mean, rstd) for
        # FREE (bias must be norm-absorbed/dropped for exactness); stats are constants
        # w.r.t. autograd (the norm backward's correction terms are their gradient).
        N, Ci, D, H, W = x.shape
        Co, _, k, _, _ = w.shape
        xm = x.permute(0, 2, 3, 4, 1)
        xm = (xm if xm.is_contiguous() else xm.contiguous()).reshape(-1, Ci)
        if i8packed is None:
            sx = _amax_scale(xm)
            x8 = quantize_fp8(xm, sx)      # one-pass Triton, no f32 materialization
            if packed is None:
                w8, sw = pack_weight_fp8(w)
                wg8, swg = (None, None) if stride != 1 else pack_weight_dgrad_fp8(w)
            else:
                w8, sw, wg8, swg = packed
        if i8packed is not None:
            # FULL int8 lane (int8-QAT): fwd AND bwd run int8 kernels (int32 acc) —
            # deploy-exact forward, no duplicate fp8 quantize, and on Ampere this is
            # the only tensor-core-fast backward. dy is int8 too (measured: uniform
            # levels beat e4m3 on normalized tensors; e2e loss curve is the gate).
            wi8, swi, wgi8, swgi, wsumv = i8packed
            if torch.is_grad_enabled():
                # DEAD BRANCH (measured 2026-07-07): grad mode is always OFF inside
                # Function.forward, so training fwd takes the dynamic-AFFINE path
                # below — which matches deploy math. Kept for documentation; if
                # torch semantics ever change this recomputes its own scale.
                sxi = _amax_scale(xm) * (E4M3_MAX / 127.0)   # amax/127
                xi8 = quantize_i8_fused(xm, sxi)
                xfi = Fp8Tensor(xi8, sxi, (N, D, H, W, Ci))
            elif i8cal is not None:
                # steady state: FROZEN calibrated (scale, zp) — zero reduces/syncs
                sxi, xzp = i8cal
                xi8 = quantize_i8_static(xm, sxi, xzp)
                xfi = Fp8Tensor(xi8, sxi, (N, D, H, W, Ci))
                xfi.zp = xzp
            elif cal_out is not None:
                # calibration: dynamic AFFINE with host sync (records scale/zp;
                # post-lrelu skew killed symmetric int8, SD 0.744)
                xi8, sxi, xzp = quantize_i8_affine(xm)
                xfi = Fp8Tensor(xi8, sxi, (N, D, H, W, Ci))
                xfi.zp = xzp
                cal_out.append((sxi, xzp))
            else:
                # TRAINING: DELAYED-SCALING affine — quantize with LAST step's
                # observed range x headroom, observing THIS batch in the same
                # kernel pass. Forensics (1500-step KD): frozen cal misfits
                # crop-to-crop variation at any cadence (0.14-0.31) vs dynamic
                # ~0.06-0.08; the dynamic premium was the aminmax REDUCE
                # (~78 ms/step), which this removes. zp stays a device tensor
                # (ZP_PTR; exact-int32 wsum epilogue).
                if dyn_state is not None:
                    xi8, sxi, xzp_t = quantize_i8_delayed(xm, dyn_state)
                else:
                    xi8, sxi, xzp_t = quantize_i8_dyn(xm)
                xfi = Fp8Tensor(xi8, sxi, (N, D, H, W, Ci))
                xfi.zp = xzp_t
            if stats_out is not None and N == 1:
                y, mean, rstd = fp8_conv3d_v2(xfi, wi8, swi, Co, k, stride,
                                              out_dtype=torch.float16, xs_f=sxi,
                                              want_stats=True, wsum=wsumv,
                                              bsmask=bsmask)
                stats_out.append((mean.reshape(1, -1), rstd.reshape(1, -1)))
            else:
                y = fp8_conv3d_v2(xfi, wi8, swi, Co, k, stride,
                                  out_dtype=torch.float16, xs_f=sxi, wsum=wsumv,
                                  bsmask=bsmask)
            if bwd_fp8 and packed is not None:
                # noise-floor experiment: int8 fwd (deploy-exact) + FP8 backward.
                # Save fp8 activations/packs so dgrad/wgrad run the higher-
                # fidelity e4m3 path (costs one extra amax+quant of x per conv).
                sx8 = _amax_scale(xm)
                x8, sx = quantize_fp8(xm, sx8), sx8
                w8, sw, wg8, swg = packed
                is_i8_bwd = False
            else:
                x8, sx, w8, sw, wg8, swg = xi8, sxi, wi8, swi, wgi8, swgi
                is_i8_bwd = True
        else:
            xf = Fp8Tensor(x8, sx, (N, D, H, W, Ci))
            # ALL scales are 1-elem device tensors (SCALE_PTR): zero host syncs per step
            if stats_out is not None and N == 1:
                y, mean, rstd = fp8_conv3d_v2(xf, w8, sw, Co, k, stride,
                                              out_dtype=torch.float16, xs_f=sx,
                                              want_stats=True, bsmask=bsmask)
                stats_out.append((mean.reshape(1, -1), rstd.reshape(1, -1)))
            else:
                y = fp8_conv3d_v2(xf, w8, sw, Co, k, stride, out_dtype=torch.float16,
                                  xs_f=sx, bsmask=bsmask)
        if bias is not None:
            y = y + bias.view(1, -1, 1, 1, 1).to(y.dtype)
        ctx.save_for_backward(x8, sx, w8, sw, wg8, swg, bsmask)
        ctx.meta = (N, Ci, Co, D, H, W, k, stride, bias is not None,
                    i8packed is not None and (not bwd_fp8 or packed is None))
        # NO forced .contiguous(): y is a channels-last-3d-layout view — in a CL3D net
        # the next conv's permute is then free (the copies were 28% of the step).
        return y

    @staticmethod
    def backward(ctx, dy):
        x8, sx, w8, sw, wg8, swg, bsmask = ctx.saved_tensors
        N, Ci, Co, D, H, W, k, stride, has_bias, is_i8 = ctx.meta
        Do, Ho, Wo = dy.shape[2], dy.shape[3], dy.shape[4]
        dym = dy.permute(0, 2, 3, 4, 1)
        dym = (dym if dym.is_contiguous() else dym.contiguous()).reshape(-1, Co)
        if is_i8:
            sdy = _amax_scale(dym) * (E4M3_MAX / 127.0)
            dy8 = quantize_i8_fused(dym, sdy)
        else:
            sdy = _amax_scale(dym)
            dy8 = quantize_fp8(dym, sdy)
        if not ctx.needs_input_grad[0]:
            dx = None            # stem conv: full-res dx for a requires_grad=False leaf
        elif stride == 1:
            # dgrad = forward conv of dy with mirrored/channel-swapped weights
            dyf = Fp8Tensor(dy8, sdy, (N, D, H, W, Co))
            dx = fp8_conv3d_v2(dyf, wg8, swg, Ci, k, 1, out_dtype=torch.float16,
                               xs_f=sdy)
        else:
            # fractionally-strided dgrad, FORWARD weight pack. int8: use the
            # per-TENSOR repack from the dgrad cache slots — Co is the reduction
            # axis in dgrad, per-channel [Co] scales cannot apply
            w_s2, sw_s2 = (wg8, swg) if is_i8 else (w8, sw)
            dx = fp8_conv3d_dgrad_s2(dy8, w_s2, (N, D, H, W, Do, Ho, Wo), Ci, Co, k,
                                     sdy, sw_s2)
        dw = _wgrad_maybe_stream(dy8, x8, (N, D, H, W, Do, Ho, Wo), Ci, Co, k, stride,
                                 sdy, sx, bsmask)
        # sum the contiguous [M,C] view with a fused-dtype reduce — dy.float() was a
        # full f32 materialization (measured ~13 -> 2.4 ms across the net's dy shapes)
        db = dym.sum(0, dtype=torch.float32) if has_bias else None
        return (None if dx is None else dx.to(dy.dtype), dw, db, None, None, None,
                None, None, None, None, None, None)


def fp8_conv3d_train(x, w, bias=None, stride=1, packed=None, stats_out=None,
                     i8packed=None, i8cal=None, cal_out=None, bsmask=None,
                     dyn_state=None, bwd_fp8=False):
    return Fp8Conv3d.apply(x, w, bias, stride, packed, stats_out, i8packed, i8cal,
                           cal_out, bsmask, dyn_state, bwd_fp8)


def _pad16(t):
    """Zero-pad dim0 to a multiple of 16 (torch._scaled_mm shape requirement).
    Zero rows contribute zero — sliced off / summed away harmlessly. torch.cat, not
    F.pad: constant_pad has no fp8 kernel."""
    r = (-t.shape[0]) % 16
    if not r:
        return t
    return torch.cat([t, torch.zeros(r, t.shape[1], dtype=t.dtype, device=t.device)])


def _smm(a8, b8_colmajor, sa, sb):
    return torch._scaled_mm(a8, b8_colmajor, scale_a=sa, scale_b=sb,
                            out_dtype=torch.float16)


def pack_transp_fp8(w):
    """w [Ci,Co,2,2,2] float -> (w8t [8Co,Ci], w8 [Ci,8Co], scale [1]) — both GEMM
    orientations (fwd uses w8t.t() col-major, dgrad uses w8.t() col-major)."""
    Ci, Co = w.shape[0], w.shape[1]
    wm = w.permute(0, 2, 3, 4, 1).reshape(Ci, 8 * Co)
    sw = (wm.abs().amax().float().clamp(min=1e-12) / E4M3_MAX).reshape(1)
    w8 = _q(wm.contiguous().float(), sw)
    return _q(wm.t().contiguous().float(), sw), w8, sw


class Fp8TranspConv3d(torch.autograd.Function):
    """ConvTranspose3d k2s2 (same-shape-doubling upsampler), all-GEMM fp8 path.
    Layout: wm[Cin, 8*Co] with 8Co index = ((i*2+j)*2+l)*Co + co for tap (i,j,l);
    y voxel (2d+i, 2h+j, 2w+l) = tap (i,j,l) of input voxel (d,h,w)."""

    @staticmethod
    def forward(ctx, x, w, bias, packed=None):
        N, Ci, D, H, W = x.shape
        Co = w.shape[1]
        xm = x.permute(0, 2, 3, 4, 1)
        xm = (xm if xm.is_contiguous() else xm.contiguous()).reshape(-1, Ci)
        sx = _amax_scale(xm)
        x8 = quantize_fp8(xm, sx)
        if packed is None:
            packed = pack_transp_fp8(w)
        w8t, w8, sw = packed                                  # [8Co,Ci], [Ci,8Co], [1]
        yg = _smm(_pad16(x8), w8t.t(), sx, sw)[:x8.shape[0]]  # [M, 8Co]
        # pixel-shuffle INTO channels-last layout ([N,2D,2H,2W,Co]-contiguous), viewed
        # as NCDHW — same one copy as before, but the CL chain survives the decoder.
        y = (yg.reshape(N, D, H, W, 2, 2, 2, Co)
               .permute(0, 1, 4, 2, 5, 3, 6, 7)
               .reshape(N, 2 * D, 2 * H, 2 * W, Co)
               .permute(0, 4, 1, 2, 3))
        if bias is not None:
            y = y + bias.view(1, -1, 1, 1, 1).to(y.dtype)
        ctx.save_for_backward(x8, sx, w8, sw)
        ctx.meta = (N, Ci, Co, D, H, W, bias is not None)
        return y

    @staticmethod
    def backward(ctx, dy):
        x8, sx, w8, sw = ctx.saved_tensors
        N, Ci, Co, D, H, W, has_bias = ctx.meta
        # inverse pixel-shuffle: [N,Co,2D,2H,2W] -> [M, 8Co] in the wm tap order
        dyg = (dy.reshape(N, Co, D, 2, H, 2, W, 2)
                 .permute(0, 2, 4, 6, 3, 5, 7, 1)
                 .reshape(-1, 8 * Co).contiguous())
        sdy = _amax_scale(dyg)
        dy8 = quantize_fp8(dyg, sdy)
        # dx_g[M,Ci] = dy_g @ wm^T : B col-major [8Co,Ci] = ([Ci,8Co] row-major).t()
        dxg = _smm(_pad16(dy8), w8.t(), sdy, sw)[:dy8.shape[0]]
        dx = dxg.reshape(N, D, H, W, Ci).permute(0, 4, 1, 2, 3).to(dy.dtype)
        # dW^T[8Co,Ci] = dy_g^T @ x_g : transpose-contiguous both (K=M padded to 16)
        dyt8 = _pad16(dy8).t().contiguous()                        # [8Co, Mp]
        xt8 = _pad16(x8).t().contiguous()                          # [Ci, Mp]
        dwt = _smm(dyt8, xt8.t(), sdy, sx)                         # [8Co, Ci]
        dw = (dwt.t().reshape(Ci, 2, 2, 2, Co).permute(0, 4, 1, 2, 3)
                 .contiguous().float())
        # dyg is [M, 8*Co] with column ((i*2+j)*2+l)*Co + co — fold the 8 tap copies
        db = dyg.sum(0, dtype=torch.float32).view(8, -1).sum(0) if has_bias else None
        return dx, dw, db, None


def fp8_transpconv3d_train(x, w, bias=None, packed=None):
    return Fp8TranspConv3d.apply(x, w, bias, packed)


class Fp8Conv3dLayer(torch.nn.Module):
    """Drop-in replacement for an eligible nn.Conv3d (k in {1,3}, same-pad, groups=1),
    running the fp8 fwd+bwd path. Weights/bias stay f32 master copies (optimizer-friendly).
    Quantize-on-update: packed fp8 weights are cached keyed on weight._version, which
    the optimizer's in-place update bumps — repack happens once per step, not per call
    (fwd/bwd share it; was 2 packs/call)."""

    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride[0]
        self._cache = (None, None)
        self._i8cache = (None, None)
        self.graph_mode = False
        self.int8_fwd = False       # int8-QAT: fwd = deployment int8 kernel, bwd = fp8
        self.i8_cal = None          # frozen (scale, zp) after int8_calibrate()
        self.i8_calibrating = False
        self._i8_obs = []
        self.emit_stats = False     # set by swap_norms_fp8 when a bound norm follows
        self.stats = None           # one-shot (mean, rstd) handoff to the bound norm
        self._dyn = {}              # delayed-scaling range state (device tensors)

    def _masked_w(self):
        wf = self.weight.detach().float()
        if getattr(self, "sparse24", False):
            wf = wf * self._mask24
        if getattr(self, "bsparse", False):
            wf = wf * self._bsmask_w
        return wf

    def _packed(self):
        v = self.weight._version
        if self._cache[0] != v:
            wf = self._masked_w()
            w8, sw = pack_weight_fp8(wf)
            wg8, swg = pack_weight_dgrad_fp8(wf) if self.stride == 1 else (None, None)
            self._cache = (v, (w8, sw, wg8, swg))    # scales stay device tensors: no sync
        return self._cache[1]

    @torch.no_grad()
    def _forward_prequant(self, x):
        """Resident-port inference path: input already int8 at our frozen
        (scale, zp) — no permute, no quantize. Inference-only by construction
        (producers only emit PreQuantI8 under no_grad)."""
        from fp8_conv3d_op import Fp8Tensor, fp8_conv3d_v2
        wi8, swi, _, _, wsumv = self._packed_i8()
        N, C, D, H, W = x.shape
        Co, k = self.weight.shape[0], self.weight.shape[2]
        xfi = Fp8Tensor(x.data, x.scale, (N, D, H, W, C))
        xfi.zp = x.zp
        want = self.emit_stats and self.bias is None and N == 1
        if want:
            y, mean, rstd = fp8_conv3d_v2(xfi, wi8, swi, Co, k, self.stride,
                                          out_dtype=torch.float16, xs_f=x.scale,
                                          want_stats=True, wsum=wsumv)
            self.stats = (mean.reshape(1, -1), rstd.reshape(1, -1))
        else:
            y = fp8_conv3d_v2(xfi, wi8, swi, Co, k, self.stride,
                              out_dtype=torch.float16, xs_f=x.scale, wsum=wsumv)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1, 1).to(y.dtype)
        return y

    def _packed_i8(self):
        v = self.weight._version
        if self._i8cache[0] != v:
            wf = self._masked_w()
            # refresh per-channel scales every K repacks; reuse otherwise
            self._i8_repacks = getattr(self, "_i8_repacks", 0) + 1
            reuse = (self._i8cache[0] is not None and self._i8_repacks % 16 != 0)
            old = self._i8cache[1][1] if reuse else None
            wi8, swi = pack_weight_i8(wf, scales=old)
            # stride 1: mirrored dgrad pack (per-ci scales, WS_VEC epilogue).
            # stride 2: dgrad_s2 reduces over Co, so per-channel [Co] scales are
            # impossible there — stash a PER-TENSOR forward-layout repack in the
            # dgrad slots (the old code passed the per-channel pack and the kernel
            # silently applied swi[0] to every channel — live gradient bug).
            wgi8, swgi = (pack_weight_dgrad_i8(wf) if self.stride == 1
                          else pack_weight_i8(wf, per_channel=False))
            self._i8cache = (v, (wi8, swi, wgi8, swgi, wsum_i8(wi8)))
        return self._i8cache[1]

    def forward(self, x):
        if isinstance(x, PreQuantI8):
            return self._forward_prequant(x)
        i8 = self._packed_i8() if self.int8_fwd else None
        # int8 lane included: AFFINE zp correction precedes STATS accumulation in
        # _conv3d_f8_v2, so epilogue (mean, rstd) are correct there too
        so = ([] if (self.emit_stats and self.bias is None and x.shape[0] == 1)
              else None)
        co = [] if (i8 is not None and self.i8_calibrating) else None
        w_in = self.weight.float()
        if getattr(self, "sparse24", False):
            w_in = w_in * self._mask24
        if getattr(self, "bsparse", False):
            w_in = w_in * self._bsmask_w
        y = fp8_conv3d_train(x, w_in,
                             None if self.bias is None else self.bias.float(),
                             self.stride,
                             None if self.graph_mode else self._packed(), so, i8,
                             # inference: frozen static (scale, zp). TRAINING:
                             # sync-free DYNAMIC affine (below) — frozen scales
                             # misfit crop-to-crop range variation no matter how
                             # often refrozen (measured 1500-step KD plateaus:
                             # dynamic ~0.06 vs recal-50 0.14, one-shot 0.31)
                             self.i8_cal if not torch.is_grad_enabled() else None,
                             co,
                             self._bs_colmask if getattr(self, "bsparse", False)
                             else None,
                             self._dyn if i8 is not None else None,
                             getattr(self, "bwd_fp8", False))
        self.stats = so[0] if so else None
        if co:
            self._i8_obs.append(co[0])
        return y


class Fp8TranspConv3dLayer(torch.nn.Module):
    """Drop-in for a k2s2 ConvTranspose3d, fp8 all-GEMM fwd+bwd, f32 master weights."""

    def __init__(self, tp):
        super().__init__()
        self.weight = tp.weight
        self.bias = tp.bias
        self._cache = (None, None)
        self.graph_mode = False

    def _packed(self):
        v = self.weight._version
        if self._cache[0] != v:
            self._cache = (v, pack_transp_fp8(self.weight.detach().float()))
        return self._cache[1]

    def forward(self, x):
        return fp8_transpconv3d_train(x, self.weight.float(),
                                      None if self.bias is None else self.bias.float(),
                                      None if self.graph_mode else self._packed())


_WG_STREAM = None
WGRAD_STREAM = False    # enable AFTER autotune is warm (concurrent kernels pollute the
                        # benchmark-based tuner); e2e flips it on post-warmup.


def _wg_stream():
    global _WG_STREAM
    if _WG_STREAM is None:
        _WG_STREAM = torch.cuda.Stream()
    return _WG_STREAM


def set_wgrad_stream(on):
    """Side-stream wgrad drain: dgrad (MMA-dense) and wgrad (atomics/latency-bound) are
    independent — overlap them. SAFE ONLY with zero_grad(set_to_none=True) (autograd
    pointer-assigns dw, nothing reads it before the step) + join_wgrad_stream() before
    opt.step(). Not graph-capture compatible."""
    global WGRAD_STREAM
    WGRAD_STREAM = on


def join_wgrad_stream():
    if _WG_STREAM is not None:
        torch.cuda.current_stream().wait_stream(_WG_STREAM)


def _wgrad_maybe_stream(dy8, x8, shapes, Ci, Co, k, stride, sdy, sx, nmask=None):
    if not WGRAD_STREAM:
        return fp8_conv3d_wgrad(dy8, x8, shapes, Ci, Co, k, stride, sdy, sx, nmask)
    s = _wg_stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        dw = fp8_conv3d_wgrad(dy8, x8, shapes, Ci, Co, k, stride, sdy, sx, nmask)
    dy8.record_stream(s)
    x8.record_stream(s)
    sdy.record_stream(s)
    sx.record_stream(s)
    return dw


_SXH = {}


def _sxh(device):
    if device not in _SXH:
        _SXH[device] = torch.full((1,), XHAT_SCALE, dtype=torch.float32, device=device)
    return _SXH[device]


class Fp8NormAct(torch.autograd.Function):
    """InstanceNorm(affine) + LeakyReLU, layout-native ([M,C] channels-last views —
    replaces cuDNN's NCHW batch-norm round-trips + eager lrelu). Saves xhat as fp8
    (FIXED scale, xhat is unit-variance) + exact sign bits; backward is the two-pass
    in_norm_act_bwd. Numerics traps inherited from the P2.2 postmortem."""

    @staticmethod
    def forward(ctx, h, gamma, beta, neg_slope, eps, stats=None):
        N, C = h.shape[0], h.shape[1]
        sp = (h.shape[2], h.shape[3], h.shape[4])
        hm = h.permute(0, 2, 3, 4, 1)
        hm = (hm if hm.is_contiguous() else hm.contiguous()).reshape(-1, C)
        if hm.dtype != torch.float16:
            hm = hm.half()
        # stats from the producing conv's epilogue (free) when bound; else re-read
        mean, rstd = stats if stats is not None else in_stats(hm, N, eps)
        y16, xh, sgn = in_norm_act_train(hm, N, mean, rstd, gamma, beta, neg_slope)
        ctx.save_for_backward(xh, sgn, rstd, gamma)
        ctx.meta = (N, C, sp, neg_slope, h.dtype)
        return y16.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)

    @staticmethod
    def backward(ctx, dout):
        xh, sgn, rstd, gamma = ctx.saved_tensors
        N, C, sp, neg_slope, hdt = ctx.meta
        dm = dout.permute(0, 2, 3, 4, 1)
        dm = (dm if dm.is_contiguous() else dm.contiguous()).reshape(-1, C)
        if dm.dtype != torch.float16:
            dm = dm.half()
        dh, dgamma, dbeta = in_norm_act_bwd(dm, xh, sgn, _sxh(dm.device), gamma, rstd,
                                            N, neg_slope)
        dh_v = dh.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)
        return (dh_v.to(hdt) if hdt != torch.float16 else dh_v, dgamma, dbeta,
                None, None, None)


class PreQuantI8:
    """int8 activations pre-quantized at the CONSUMER conv's frozen (scale, zp)
    by the producing kernel (resident port) — the conv skips its quantize pass."""

    __slots__ = ("data", "scale", "zp", "shape")

    def __init__(self, data, scale, zp, shape):   # shape = (N, C, D, H, W)
        self.data, self.scale, self.zp, self.shape = data, scale, zp, shape


class Fp8NormActLayer(torch.nn.Module):
    def __init__(self, norm, nonlin=None, src_conv=None):
        super().__init__()
        self.gamma = norm.weight
        self.beta = norm.bias
        self.eps = norm.eps
        self.neg_slope = (nonlin.negative_slope
                          if isinstance(nonlin, torch.nn.LeakyReLU) else 1.0)
        self.src_conv = [src_conv] if src_conv is not None else None  # no submodule reg
        self.consumer = None        # [Fp8Conv3dLayer] bound by int8_calibrate

    def forward(self, h):
        stats = None
        if self.src_conv is not None:
            stats, self.src_conv[0].stats = self.src_conv[0].stats, None  # one-shot
        if not torch.is_grad_enabled():
            # INFERENCE: y16-only kernel — the train path also emits xhat-fp8 +
            # sign bits for backward (dead stores + extra bandwidth under no_grad)
            from fp8_conv3d_op import in_norm_act, in_stats
            N, C = h.shape[0], h.shape[1]
            sp = (h.shape[2], h.shape[3], h.shape[4])
            hm = h.permute(0, 2, 3, 4, 1)
            hm = (hm if hm.is_contiguous() else hm.contiguous()).reshape(-1, C)
            if hm.dtype != torch.float16:
                hm = hm.half()
            mean, rstd = stats if stats is not None else in_stats(hm, N, self.eps)
            cons = self.consumer[0] if self.consumer else None
            if cons is not None and cons.int8_fwd and cons.i8_cal is not None:
                # resident port: emit the consumer's affine int8 directly
                q = in_norm_act(hm, N, mean, rstd, self.gamma.float(),
                                self.beta.float(), self.neg_slope,
                                i8_cal=cons.i8_cal)
                return PreQuantI8(q, cons.i8_cal[0], cons.i8_cal[1],
                                  (N, C, *sp))
            y16 = in_norm_act(hm, N, mean, rstd, self.gamma.float(),
                              self.beta.float(), self.neg_slope)
            return y16.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)
        return Fp8NormAct.apply(h, self.gamma, self.beta, self.neg_slope, self.eps,
                                stats)


def swap_norms_fp8(net):
    """Replace adjacent (InstanceNorm3d(affine), LeakyReLU) pairs — and lone affine
    InstanceNorm3d — inside every Sequential/container with the layout-native fp8
    norm+act. Norms directly preceded by a bias-free Fp8Conv3dLayer sibling BIND to it:
    the conv's epilogue computes their (mean, rstd) for free (N=1; exact because the
    bias was norm-absorbed/dropped). Run AFTER swap_convs_fp8. Returns count."""
    n = 0
    for mod in list(net.modules()):
        kids = list(mod.named_children())
        for i, (cname, child) in enumerate(kids):
            if not (isinstance(child, torch.nn.InstanceNorm3d) and child.affine):
                continue
            src = None
            for j in range(i - 1, -1, -1):
                sib = kids[j][1]
                if isinstance(sib, Fp8Conv3dLayer):
                    if sib.bias is None:
                        src = sib
                        sib.emit_stats = True
                    break
                if not isinstance(sib, (torch.nn.Identity, torch.nn.Dropout3d)):
                    break
            nxt = kids[i + 1][1] if i + 1 < len(kids) else None
            if isinstance(nxt, torch.nn.LeakyReLU):
                setattr(mod, cname, Fp8NormActLayer(child, nxt, src))
                setattr(mod, kids[i + 1][0], torch.nn.Identity())
            else:
                setattr(mod, cname, Fp8NormActLayer(child, None, src))
            n += 1
    return n


class Fp8Tail(torch.autograd.Function):
    """BasicBlockD tail: y = lrelu(h*(gc+gs) + res) [scSE] or lrelu(h+res) — ONE kernel
    each direction, replacing 3-4 eager elementwise passes + their fp16 saved tensors.
    Saves h fp8 + sign bits; gate grads (dgc/dgs) flow back into the torch-side SE
    subgraph (pools + gate convs stay autograd — their params train normally)."""

    @staticmethod
    def forward(ctx, h, res, gc, gs, neg_slope):
        N, C = h.shape[0], h.shape[1]
        sp = (h.shape[2], h.shape[3], h.shape[4])
        hm = h.permute(0, 2, 3, 4, 1)
        hm = (hm if hm.is_contiguous() else hm.contiguous()).reshape(-1, C)
        rm = res.permute(0, 2, 3, 4, 1)
        rm = (rm if rm.is_contiguous() else rm.contiguous()).reshape(-1, C)
        if rm.dtype != hm.dtype:
            rm = rm.to(hm.dtype)
        has_se = gc is not None
        if has_se:
            sh = _amax_scale(hm)
            gcf = gc.reshape(N, C).float().contiguous()
            gsf = gs.reshape(-1).contiguous()
            y16, h8, sgn = tail_train_fwd(hm, rm, gcf, gsf, N, neg_slope,
                                          torch.reciprocal(sh))
        else:
            sh = gcf = gsf = h8 = None
            y16, _, sgn = tail_train_fwd(hm, rm, None, None, N, neg_slope, None)
        ctx.save_for_backward(h8, sgn, gcf, gsf, sh)
        ctx.meta = (N, C, sp, neg_slope, has_se, h.dtype,
                    None if gs is None else gs.dtype)
        return y16.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)

    @staticmethod
    def backward(ctx, dout):
        h8, sgn, gcf, gsf, sh = ctx.saved_tensors
        N, C, sp, neg_slope, has_se, hdt, gsdt = ctx.meta
        dm = dout.permute(0, 2, 3, 4, 1)
        dm = (dm if dm.is_contiguous() else dm.contiguous()).reshape(-1, C)
        if dm.dtype != torch.float16:
            dm = dm.half()
        dh, dres, dgc, dgs = tail_train_bwd(dm, h8, sgn, gcf, gsf, sh, N, neg_slope)
        dh_v = dh.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)
        dres_v = dres.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)
        if has_se:
            dgc = dgc.reshape(N, C, 1, 1, 1)
            dgs = dgs.reshape(N, 1, *sp).to(gsdt)
        return (dh_v.to(hdt) if hdt != torch.float16 else dh_v, dres_v,
                dgc, dgs, None)


def _fp8_block_forward(self, x):
    residual = self.skip(x)
    out = self.conv2(self.conv1(x))
    gc = gs = None
    if self.apply_se:
        se = self.squeeze_excitation
        s = out.mean((2, 3, 4), keepdim=True)
        gc = se.cSE.gate(se.cSE.fc2(se.cSE.act(se.cSE.fc1(s))))    # [N,C,1,1,1]
        gs = se.sSE.gate(se.sSE.conv(out))                          # [N,1,D,H,W]
    if not torch.is_grad_enabled() and out.shape[0] == 1:
        # INFERENCE tail: y16-only kernel — no h8/sign stores, no sh amax
        from fp8_conv3d_op import block_tail
        N, C = out.shape[0], out.shape[1]
        sp = (out.shape[2], out.shape[3], out.shape[4])
        hm = out.permute(0, 2, 3, 4, 1)
        hm = (hm if hm.is_contiguous() else hm.contiguous()).reshape(-1, C)
        rm = residual.permute(0, 2, 3, 4, 1)
        rm = (rm if rm.is_contiguous() else rm.contiguous()).reshape(-1, C)
        if rm.dtype != hm.dtype:
            rm = rm.to(hm.dtype)
        gcf = gc.reshape(C).float().contiguous() if gc is not None else None
        gsf = gs.reshape(-1).float().contiguous() if gs is not None else None
        y16, _ = block_tail(hm, rm, gcf, gsf, self.nonlin2.negative_slope)
        return y16.reshape(N, *sp, C).permute(0, 4, 1, 2, 3)
    return Fp8Tail.apply(out, residual, gc, gs, self.nonlin2.negative_slope)


def _first_conv(stage):
    root = getattr(stage, "all_modules", stage)
    for m in root.modules():
        if isinstance(m, Fp8Conv3dLayer):
            return m
    return None


def _fs_decoder_body(self, skips):
    from fp8_conv3d_op import quant_cat2_i8
    lres = skips[-1]
    for s in range(len(self.stages)):
        x = self.transpconvs[s](lres)
        skip = skips[-(s + 2)]
        conv = _first_conv(self.stages[s].convs[0]) if hasattr(
            self.stages[s], "convs") else _first_conv(self.stages[s])
        if (not torch.is_grad_enabled()) and conv is not None \
                and conv.int8_fwd and conv.i8_cal is not None:
            # int8 cat fusion: cat+quantize in one kernel at the stage conv's
            # frozen affine cal; the conv takes the PreQuantI8 fast path
            N, CA = x.shape[0], x.shape[1]
            CB = skip.shape[1]
            sp = (x.shape[2], x.shape[3], x.shape[4])
            am = x.permute(0, 2, 3, 4, 1)
            am = (am if am.is_contiguous() else am.contiguous()).reshape(-1, CA)
            bm = skip.permute(0, 2, 3, 4, 1)
            bm = (bm if bm.is_contiguous() else bm.contiguous()).reshape(-1, CB)
            if am.dtype != torch.float16:
                am = am.half()
            if bm.dtype != torch.float16:
                bm = bm.half()
            q = quant_cat2_i8(am, bm, conv.i8_cal)
            x = PreQuantI8(q, conv.i8_cal[0], conv.i8_cal[1],
                           (N, CA + CB, *sp))
        else:
            x = torch.cat((x, skip), 1)
        x = self.stages[s](x)
        lres = x
    return lres


def _fs_decoder_forward(self, skips):
    return self.seg_layers[-1](_fs_decoder_body(self, skips))


def swap_decoder_cats(net):
    """Bind the int8 cat-fusion decoder forward onto reference.Decoder /
    DecoderBody instances (inference-only dispatch inside). Returns count."""
    import types as _types
    n = 0
    for m in net.modules():
        if hasattr(m, "transpconvs") and hasattr(m, "stages"):
            if hasattr(m, "seg_layers"):
                m.forward = _types.MethodType(_fs_decoder_forward, m)
            else:
                m.forward = _types.MethodType(_fs_decoder_body, m)
            n += 1
    return n


def swap_tails_fp8(net):
    """Bind the fused-tail forward onto every BasicBlockD instance (scSE-apply +
    residual + lrelu in one kernel each direction). Composes with swap_convs_fp8 +
    swap_norms_fp8. Returns count."""
    import types as _types
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD
    n = 0
    for m in net.modules():
        if isinstance(m, BasicBlockD) and not m.apply_stochastic_depth:
            m.forward = _types.MethodType(_fp8_block_forward, m)
            n += 1
    swap_decoder_cats(net)   # int8 cat fusion (inference-only dispatch inside)
    return n


def mask_24(w):
    """2:4 structured-sparsity mask (top-2 magnitude per group of 4 along the flattened
    reduction dim of the [Co, K] view — torch-canonical flatten, the ASP convention).
    NOTE for TRT export: TRT validates 2:4 in ITS weight layout; re-check group axis at
    engine-build time (kSPARSE_WEIGHTS refuses non-conforming weights loudly)."""
    Co = w.shape[0]
    flat = w.detach().abs().reshape(Co, -1)
    K = flat.shape[1]
    if K % 4 != 0:
        return torch.ones_like(w, dtype=torch.bool)
    g = flat.reshape(Co, K // 4, 4)
    idx = g.topk(2, dim=2).indices
    m = torch.zeros_like(g, dtype=torch.bool)
    m.scatter_(2, idx, True)
    return m.reshape(w.shape)


def set_blocksparse(net, ratio=0.5, group=32):
    """BLOCK sparsity experiment: prune the lowest-L2 (tap, cin)-GROUPS of each conv's
    packed K axis (32-wide, matching the kernels' skip granularity), globally per layer
    across all output channels. Unlike 2:4 (TRT-only), OUR Triton kernels skip pruned
    chunks: fwd conv K-loop and wgrad N-blocks — real measured speedup, exact math
    (pruned weights are literal zeros). Coarser than 2:4 -> accuracy is the experiment;
    the KD/QAT loop is the recovery mechanism. Returns (layers, actual_sparsity)."""
    n, pruned, total = 0, 0, 0
    for m in net.modules():
        if not isinstance(m, Fp8Conv3dLayer):
            continue
        w = m.weight.detach().float()
        Co, Ci = w.shape[0], w.shape[1]
        k3 = w.shape[2] * w.shape[3] * w.shape[4]
        K = k3 * Ci
        if K % group or K // group < 4:
            continue
        # pack-layout columns: col = tap*Ci + ci  -> group norms over [Co, group]
        cols = w.reshape(Co, Ci, k3).permute(0, 2, 1).reshape(Co, K)
        gn = cols.reshape(Co, K // group, group).float().pow(2).sum((0, 2)).sqrt()
        nprune = int(round((K // group) * ratio))
        prune_idx = gn.argsort()[:nprune]
        colmask = torch.ones(K // group, dtype=torch.bool, device=w.device)
        colmask[prune_idx] = False
        m._bs_colmask = colmask.to(torch.int8).contiguous()      # kernel skip table
        wm = (colmask.repeat_interleave(group).reshape(k3, Ci)
              .permute(1, 0).reshape(1, Ci, *w.shape[2:]))       # weight-shape mask
        m._bsmask_w = wm.expand_as(w).contiguous()
        m.bsparse = True
        m._cache = (None, None)
        m._i8cache = (None, None)
        pruned += nprune * group * Co
        total += K * Co
        n += 1
    return n, pruned / max(total, 1)


def set_sparse24(net, on=True):
    """Sparse-QAT: enforce a FIXED 2:4 mask (computed from current magnitudes, the ASP
    recipe) on every Fp8Conv3dLayer at pack time — fwd AND bwd see masked weights
    (consistent function), gradients flow to ALL weights (straight-through), the mask
    holds the pruned slots at zero in every quantized pack. Composes with --int8qat:
    the student trains toward TRT sparse-int8 deployment (up to 4x dense-fp16 on
    Ampere sparse tensor cores). Returns (layers, actual_sparsity)."""
    n, zeros, total = 0, 0, 0
    for m in net.modules():
        if isinstance(m, Fp8Conv3dLayer):
            if on:
                m._mask24 = mask_24(m.weight)
                zeros += int((~m._mask24).sum())
                total += m._mask24.numel()
            m.sparse24 = on
            m._cache = (None, None)
            m._i8cache = (None, None)
            n += 1
    return n, (zeros / max(total, 1))


def set_int8_bwd_fp8(net, on=True):
    """Noise-floor experiment: int8 fwd (deploy-exact) + fp8 backward on sm120.
    Costs one extra amax+fp8-quant of x per conv; measures whether the int8-QAT
    KD plateau (~0.06 vs fp16 twin 0.01) is dy-int8 gradient noise."""
    n = 0
    for m in net.modules():
        if isinstance(m, Fp8Conv3dLayer) and m.int8_fwd:
            m.bwd_fp8 = on
            n += 1
    return n


def int8_calibrate(net, patches):
    """Static int8 activation calibration: run the given patches with dynamic affine
    quant recording per-layer (scale, zp); freeze the max-range observation. After
    this, inference quantizes with ZERO reduces/syncs (quantize_i8_static)."""
    layers = [m for m in net.modules()
              if isinstance(m, Fp8Conv3dLayer) and m.int8_fwd]
    for m in layers:
        m.i8_calibrating, m._i8_obs, m.i8_cal = True, [], None
    with torch.no_grad():
        for p in patches:
            net(p)
    for m in layers:
        if m._i8_obs:
            # UNION of observed ranges, not widest-scale-wins: (scale, zp) pairs
            # with different zp cover different [mn, mx] intervals — max scale
            # does NOT contain them all. Reconstruct each interval, union, and
            # re-derive (scale, zp). Multi-patch calibration matters: frozen
            # scales misfit crop-to-crop range variation (measured).
            mns, mxs = [], []
            for sc, zp in m._i8_obs:
                scf = float(sc)
                mns.append(-(zp + 127.0) * scf)
                mxs.append((127.0 - zp) * scf)
            mn, mx = min(min(mns), 0.0), max(max(mxs), 0.0)
            scale = torch.tensor([max((mx - mn) / 254.0, 1e-12)],
                                 device=m._i8_obs[0][0].device)
            zp = int(round(-mn / float(scale))) - 127
            m.i8_cal = (scale, max(-127, min(127, zp)))
        m.i8_calibrating = False
        m._i8_obs = []
    _bind_consumers(net)
    return len(layers)


def _bind_consumers(net, patch=None):
    """STRUCTURAL norm->conv consumer binding (resident port). Only provably
    single-consumer edges qualify — a traced-identity approach was tried and is
    unsound: non-module consumers (the BasicBlockD identity skip, SE gates) share
    the same tensor object invisibly, and Python id() reuse after GC fakes hits.
    Safe edges: CDNR[i].norm -> CDNR[i+1].conv inside a conv stack (the stack's
    LAST norm feeds outward — excluded), and conv1.norm -> conv2.conv inside
    BasicBlockD (conv2's norm feeds tail+SE — excluded)."""
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD

    def _find(mod, cls):
        # CDNR keeps alias children (.conv/.norm) NEXT TO .all_modules; the
        # swaps produce twin instances and only the all_modules one runs —
        # search there or a dead twin (never calibrated) gets bound
        root = getattr(mod, "all_modules", mod)
        for m in root.modules():
            if isinstance(m, cls):
                return m
        return None

    n_bound = 0
    for m in net.modules():
        pairs = []
        if isinstance(m, BasicBlockD):
            pairs.append((m.conv1, m.conv2))
        elif hasattr(m, "convs") and isinstance(m.convs, torch.nn.Sequential):
            cdnrs = list(m.convs)
            pairs += list(zip(cdnrs[:-1], cdnrs[1:]))
        for prod, cons in pairs:
            norm = _find(prod, Fp8NormActLayer)
            conv = _find(cons, Fp8Conv3dLayer)
            if norm is not None and conv is not None and conv.int8_fwd \
                    and conv.i8_cal is not None:
                norm.consumer = [conv]
                n_bound += 1
    return n_bound


def set_int8_qat(net, on=True):
    """int8-QAT: every Fp8Conv3dLayer's FORWARD runs the deployment int8 kernel
    (train-time fwd == deploy-time int8 fwd, same kernel + layout); backward stays
    the validated fp8 dgrad/wgrad — a straight-through estimate through the int8
    quantizer (which measured MORE accurate than fp8: depth-8 corr 0.99853)."""
    n = 0
    for m in net.modules():
        if isinstance(m, Fp8Conv3dLayer):
            m.int8_fwd = on
            n += 1
    return n


def set_graph_mode(net, on=True):
    """CUDA-graph-capturable packing: bypass the host-side weight._version cache and
    pack weights INSIDE the autograd Function — the captured pack kernels re-quantize
    from the in-place-updated master weights on every replay (host cache logic never
    runs during replay, so it would serve stale packs)."""
    n = 0
    for m in net.modules():
        if hasattr(m, "graph_mode"):        # Fp8Conv3dLayer/Fp8TranspConv3dLayer/Fp8CDNRLayer
            m.graph_mode = on
            n += 1
    return n


class ClAvgPool2(torch.nn.Module):
    """channels-last 2^3 avg pool (Triton, fwd+bwd) — drop-in for the skip-path
    AvgPool3d(2,2); output is a CL view feeding the skip 1x1 Fp8Conv3dLayer."""

    def forward(self, x):
        from fp8_conv3d_op import avgpool2_cl
        return avgpool2_cl(x)


class CastConv3d(torch.nn.Module):
    """Plain conv3d with f32 master params cast to the input dtype per call — used for
    SE-internal convs (kept out of fp8: M=1/Co=1 shapes are tensor-core-hostile and
    were the grad-corr tail), which now see fp16 activations from the fp8 layers."""

    def __init__(self, conv):
        super().__init__()
        self.weight = conv.weight
        self.bias = conv.bias
        self.stride = conv.stride
        self.padding = conv.padding

    def forward(self, x):
        return torch.nn.functional.conv3d(
            x, self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
            self.stride, self.padding)


def swap_convs_fp8(net):
    """Replace every eligible nn.Conv3d with Fp8Conv3dLayer and every k2s2
    ConvTranspose3d with Fp8TranspConv3dLayer, in-place. Returns (swapped, kept).
    Conv eligible: k in {1,3}, same-pad, stride in {1,2}, groups=1, 5D.
    SE-internal convs stay fp16 (M=1/Co=1 shapes: tensor-core-hostile AND the source
    of the low grad-corr tail). Convs directly followed by InstanceNorm drop their
    bias (exact absorption — per-channel constants cancel in the norm's mean; grads
    were pure noise). MUST run before swap_norms_fp8 (norm detection)."""
    swapped = kept = 0
    for name, mod in list(net.named_modules()):
        if "squeeze_excitation" in name:
            for cname, c in list(mod.named_children()):
                if isinstance(c, torch.nn.Conv3d):
                    setattr(mod, cname, CastConv3d(c))
                    kept += 1
            continue
        kids = list(mod.named_children())
        for i, (cname, child) in enumerate(kids):
            if isinstance(child, torch.nn.AvgPool3d) and \
                    child.kernel_size in (2, (2, 2, 2)) and \
                    child.stride in (2, (2, 2, 2)):
                # BasicBlockD stride-2 skip: CL-native pool (AvgPool3d forces a
                # full NCDHW contiguous on the channels-last view, ~6.5 ms/patch)
                setattr(mod, cname, ClAvgPool2())
                continue
            if isinstance(child, torch.nn.ConvTranspose3d):
                ok = (child.kernel_size == (2, 2, 2) and child.stride == (2, 2, 2)
                      and child.groups == 1 and child.padding == (0, 0, 0)
                      and child.output_padding == (0, 0, 0)
                      and child.dilation == (1, 1, 1))
                if ok:
                    setattr(mod, cname, Fp8TranspConv3dLayer(child))
                    swapped += 1
                else:
                    kept += 1
            elif isinstance(child, torch.nn.Conv3d):
                k = child.kernel_size
                ok = (k[0] == k[1] == k[2] and k[0] in (1, 3) and child.groups == 1
                      and child.stride[0] == child.stride[1] == child.stride[2]
                      and child.stride[0] in (1, 2)
                      and child.padding == ((k[0] - 1) // 2,) * 3
                      and child.dilation == (1, 1, 1))
                if ok:
                    absorbed = (isinstance(getattr(mod, "norm", None),
                                           torch.nn.InstanceNorm3d)
                                or any(isinstance(kids[j][1], torch.nn.InstanceNorm3d)
                                       for j in range(i + 1, len(kids))))
                    lay = Fp8Conv3dLayer(child)
                    if absorbed and lay.bias is not None:
                        lay.bias.requires_grad_(False)
                        lay.bias = None
                    setattr(mod, cname, lay)
                    swapped += 1
                else:
                    kept += 1
    return swapped, kept


def corr(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


def bench(fn, it=20, wu=5):
    for _ in range(wu):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=48)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(0)
    P = args.patch

    print("== grad correctness vs f32 autograd ==")
    for C in (32, 64, 128):
        x = (torch.randn(1, C, P, P, P, device=dev) * 0.5).requires_grad_(True)
        w = (torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)).requires_grad_(True)
        gy = torch.randn(1, C, P, P, P, device=dev) * 0.1

        y = fp8_conv3d_train(x, w)
        y.backward(gy)
        dx8, dw8 = x.grad.clone(), w.grad.clone()
        x.grad = w.grad = None

        yr = F.conv3d(x.float(), w.float(), padding=1)
        yr.backward(gy.float())
        dxr, dwr = x.grad.clone(), w.grad.clone()
        x.grad = w.grad = None

        print(f"  C={C:>3}: y corr={corr(y, yr):.4f}  dx corr={corr(dx8, dxr):.4f}  "
              f"dw corr={corr(dw8, dwr):.4f}")

    print("\n== transpconv k2s2 grad correctness vs f32 autograd ==")
    for Ci, Co, Pd in ((320, 256, 4), (64, 32, P // 2)):
        x = (torch.randn(1, Ci, Pd, Pd, Pd, device=dev) * 0.5).requires_grad_(True)
        w = (torch.randn(Ci, Co, 2, 2, 2, device=dev) * (1.0 / (Ci * 8) ** 0.5)).requires_grad_(True)
        b = torch.randn(Co, device=dev).requires_grad_(True)
        gy = torch.randn(1, Co, 2 * Pd, 2 * Pd, 2 * Pd, device=dev) * 0.1

        y = fp8_transpconv3d_train(x, w, b)
        y.backward(gy)
        dx8, dw8, db8 = x.grad.clone(), w.grad.clone(), b.grad.clone()
        x.grad = w.grad = b.grad = None

        yr = F.conv_transpose3d(x.float(), w.float(), b.float(), stride=2)
        yr.backward(gy.float())
        dxr, dwr, dbr = x.grad.clone(), w.grad.clone(), b.grad.clone()
        x.grad = w.grad = b.grad = None

        print(f"  Ci={Ci:>3} Co={Co:>3} in={Pd}^3: y corr={corr(y, yr):.4f}  "
              f"dx corr={corr(dx8, dxr):.4f}  dw corr={corr(dw8, dwr):.4f}  "
              f"db corr={corr(db8, dbr):.4f}")

    print("\n== fwd+bwd micro-bench vs fp16 autograd (per call) ==")
    for C in (64, 128):
        x = (torch.randn(1, C, P, P, P, device=dev) * 0.5)
        w = (torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5))
        gy = torch.randn(1, C, P, P, P, device=dev) * 0.1

        def f8():
            xr = x.detach().requires_grad_(True)
            wr = w.detach().requires_grad_(True)
            fp8_conv3d_train(xr, wr).backward(gy)

        xh, wh, gh = x.half(), w.half(), gy.half()

        def f16():
            xr = xh.detach().requires_grad_(True)
            wr = wh.detach().requires_grad_(True)
            F.conv3d(xr, wr, padding=1).backward(gh)

        t8, t16 = bench(f8), bench(f16)
        print(f"  C={C:>3}: fp8 {t8:6.1f}ms  fp16-cuDNN {t16:6.1f}ms  speedup {t16/t8:.2f}x")

    if args.steps:
        print(f"\n== optimizer sanity ({args.steps} steps, must decrease) ==")
        C = 64
        x = torch.randn(1, C, P, P, P, device=dev) * 0.5
        wt = torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)  # target
        target = F.conv3d(x, wt, padding=1)
        w = (torch.randn(C, C, 3, 3, 3, device=dev) * (1.0 / (C * 27) ** 0.5)).requires_grad_(True)
        opt = torch.optim.Adam([w], lr=3e-3)
        for i in range(args.steps):
            opt.zero_grad()
            loss = F.mse_loss(fp8_conv3d_train(x, w).float(), target)
            loss.backward()
            opt.step()
            if i % max(1, args.steps // 6) == 0 or i == args.steps - 1:
                print(f"  step {i:>3}: loss {loss.item():.5f}")


if __name__ == "__main__":
    main()
