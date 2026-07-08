"""Whole-net fp8 forward + end-to-end accuracy gate for the surface ResEnc-UNet (sm120).

The building-block spike (fp8_conv3d_op.py, fp8_depth_real.py) proved fused fp8 conv3d is
2.85-4.14x faster than cuDNN fp16 and holds corr >=0.9996 over the deepest stage with real
trained weights. This closes the loop: run the ENTIRE reference net with its expensive
3x3x3 convs routed through the fp8 op, on a real CT patch, and compare the surface
probability output against the pure-fp16 reference.

Rather than reimplement the net, we monkeypatch nn.Conv3d.forward: any 3x3x3 conv (the
compute-dominant ones, encoder + decoder stages) dispatches to fp8_conv3d; everything else
(1x1x1 SE/seg heads, the stem, transpose-convs) stays in torch fp16. This mirrors how a real
fp8-resident net ships — quantize the big convs, leave the cheap/sensitive ops alone.

InstanceNorm + residual adds stay f32 (sensitive-path recipe). The conv is done fp8 for the
matmul, dequantized immediately (the surrounding module still expects a normal tensor), so
this measures the fp8 MATMUL error end-to-end without needing to thread Fp8Tensor through
upstream nnU-Net modules. (A production Fp8Net keeps activations fp8-resident to also capture
the speed; here we isolate the ACCURACY question.)

Gate: SurfaceDice@2 vs fp16 reference must be >= 0.998 (ADR 0010 tolerance bar). Also reports
corr and the fp16-vs-fp8 conv MATMUL wall-clock ratio for the 3x3x3 layers.

Usage: python fp8_forward.py [--ct <path.fxvol|.npy>] [--patch 128]
"""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fp8_conv3d_op import (Fp8Tensor, block_tail, pack_weight_fp8, fp8_conv3d,
                           fp8_conv3d_v2, fused_norm_act, quant_cat2, quant_cat_shuffle,
                           quantize_fp8)

_WSF = {}   # (id(conv), fine) -> python float weight scale

_orig_conv3d_fwd = nn.Conv3d.forward
_FP8_STATS = {"fp8_calls": 0, "fallback_calls": 0}
# Modules explicitly excluded from fp8 (precision-sensitive layer selection). Populated by
# mark_fp16_modules() with the last-N decoder-stage convs (nearest the seg head).
_FP16_KEEP = set()
_FINE_SCALE = False   # toggled by --fine-scale: per-channel weight + per-token activation


def _fp8_conv3d_forward(self, x):
    """Route eligible 3x3x3 stride-conv through fp8; else fall back to torch."""
    k = self.kernel_size
    eligible = (k == (3, 3, 3) and self.groups == 1 and self.padding == (1, 1, 1)
                and id(self) not in _FP16_KEEP
                and x.dtype in (torch.float16, torch.float32, torch.bfloat16))
    if not eligible:
        # still CALIBRATE 1x1 convs (skip projections): resident mode runs them in fp8, and
        # their scale must come from a clean calibration forward, not on-the-fly.
        if (_CALIBRATING and k == (1, 1, 1) and self.groups == 1 and x.dim() == 5
                and x.dtype in (torch.float16, torch.float32, torch.bfloat16)):
            s = (x.float().abs().amax().clamp(min=1e-8) / 448.0).reshape(1)
            kk = id(self)
            _ACAL[kk] = torch.maximum(_ACAL[kk], s) if kk in _ACAL else s
        _FP8_STATS["fallback_calls"] += 1
        return _orig_conv3d_fwd(self, x)
    _FP8_STATS["fp8_calls"] += 1
    stride = self.stride[0]
    Cout = self.weight.shape[0]
    # REAL Triton kernel, fp16 output (each conv feeds InstanceNorm next — matches autocast).
    # --fine-scale = per-token act + per-out-channel weight scales in-kernel.
    xf = _quantize_input(self, x, _FINE_SCALE)
    w_fp8, w_scale = _packed_weight(self, _FINE_SCALE)
    out = fp8_conv3d(xf, w_fp8, w_scale, Cout, 3, stride, out_scale=None,
                     out_dtype=torch.float16)
    if self.bias is not None:
        out = out + self.bias.data.view(1, -1, 1, 1, 1).to(out.dtype)
    return out.to(x.dtype)


_WCACHE = {}
_ACAL = {}          # id(mod) -> f32[1] static input scale (per-tensor mode)
_CALIBRATING = True  # first forward records amax; freeze_calibration() stops updates


_ACALF = {}   # id(mod) -> python float scale (no-sync access for CUDA-graph capture)


def freeze_calibration():
    global _CALIBRATING
    _CALIBRATING = False
    for k, v in _ACAL.items():
        _ACALF[k] = float(v)


def _quantize_input(mod, x, fine):
    """fp8-quantize the conv input. Per-tensor mode uses the STATIC calibrated scale after
    the first (calibration) forward — no runtime amax reduction. channels_last_3d inputs
    quantize without a permute copy."""
    N, C, D, H, W = x.shape
    xc = x.permute(0, 2, 3, 4, 1)                          # view; free if channels_last_3d
    xc = xc.reshape(N * D * H * W, C) if xc.is_contiguous() else \
        xc.contiguous().reshape(N * D * H * W, C)
    if fine:
        scale = (xc.abs().amax(dim=1).clamp(min=1e-8) / 448.0).float()
        d = (xc / scale[:, None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        return Fp8Tensor(d, scale, (N, D, H, W, C))
    k = id(mod)
    if _CALIBRATING or k not in _ACAL:
        s = (xc.abs().amax().clamp(min=1e-8) / 448.0).float().reshape(1)
        _ACAL[k] = torch.maximum(_ACAL[k], s) if k in _ACAL else s   # running max amax
    scale = _ACAL[k]
    d = (xc / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    return Fp8Tensor(d, scale, (N, D, H, W, C))


def _packed_weight(mod, fine):
    """Per-module cached fp8 weight pack (weights are static at inference)."""
    key = (id(mod), fine)
    if key not in _WCACHE:
        _WCACHE[key] = pack_weight_fp8(mod.weight.data.float(), per_channel=fine)
    return _WCACHE[key]


# ---------------- resident mode: fused conv->norm(->act)(->fp8) block execution ----------
# Replaces ConvDropoutNormReLU / BasicBlockD forwards so the fp8 chain runs with ONE fused
# glue kernel between convs (norm+act+quant) instead of 3 torch passes, static calibrated
# scales (no runtime amax), and conv bias omitted (absorbed exactly by InstanceNorm).

def _cdnr_fused(mod, x, emit_scale=None, emit_inv=None):
    """mod: ConvDropoutNormReLU (conv[+norm][+nonlin]). x: fp16 NCHW tensor | Fp8Tensor.
    Returns fp16 NCHW(-view) or, with emit_scale, an Fp8Tensor for the next fp8 conv."""
    conv = mod.conv
    k = conv.kernel_size[0]
    Cout = conv.weight.shape[0]
    if not isinstance(x, Fp8Tensor):
        assert x.shape[0] == 1, "fp8 resident path is N=1 (InstanceNorm stats are pooled " \
            "over the whole batch here; use predict-scroll gpuworkers= for parallelism)"
    if isinstance(x, Fp8Tensor):
        xf = x
    elif id(conv) in _PENDING8:
        xf = _PENDING8.pop(id(conv))    # previous block's tail already emitted our fp8 input
    else:
        key = id(conv)
        if key not in _ACAL:  # calibration miss: record on the fly
            _ACAL[key] = (x.float().abs().amax().clamp(min=1e-8) / 448.0).reshape(1)
            _ACALF[key] = float(_ACAL[key])
        N, C, D, H, W = x.shape
        xc = x.permute(0, 2, 3, 4, 1)
        xc = (xc if xc.is_contiguous() else xc.contiguous()).reshape(N * D * H * W, C)
        d = quantize_fp8(xc.half() if xc.dtype != torch.float16 else xc, _ACAL[key],
                         inv_s=1.0 / _ACALF[key])
        xf = Fp8Tensor(d, _ACAL[key], (N, D, H, W, C))
    w_fp8, w_scale = _packed_weight(conv, False)   # resident mode is per-tensor throughout
    wkey = (id(conv), False)
    if wkey not in _WSF:
        _WSF[wkey] = float(w_scale)
    h, mean, rstd = fp8_conv3d_v2(xf, w_fp8, _WSF[wkey], Cout, k, conv.stride[0],
                                  out_dtype=torch.float16,
                                  xs_f=_ACALF.get(id(conv)),
                                  want_stats=True)             # NCHW channels-last view
    N, _, Do, Ho, Wo = h.shape
    h_mc = h.permute(0, 2, 3, 4, 1).reshape(N * Do * Ho * Wo, Cout)  # storage-order view
    act = hasattr(mod, "nonlin") and isinstance(mod.nonlin, nn.LeakyReLU)
    ns = mod.nonlin.negative_slope if act else 0.01
    y = fused_norm_act(h_mc, mod.norm.weight.data.float(), mod.norm.bias.data.float(),
                       act=act, neg_slope=ns, eps=mod.norm.eps, out_fp8_scale=emit_scale,
                       inv_out=emit_inv, stats=(mean, rstd))
    if emit_scale is not None:
        return Fp8Tensor(y, emit_scale, (N, Do, Ho, Wo, Cout))
    return y.reshape(N, Do, Ho, Wo, Cout).permute(0, 4, 1, 2, 3)


_NEXT8 = {}     # id(block) -> (id of successor conv1.conv) — built by build_successor_map
_PENDING8 = {}  # id(conv) -> Fp8Tensor stashed by the previous block's fused tail


def build_successor_map(net):
    """After calibration: block[i] tail emits fp8 directly for block[i+1].conv1 (within a
    stage, and across encoder stage boundaries for stride-1-compatible inputs)."""
    _NEXT8.clear()
    try:
        stages = list(net.shared_encoder.stages)
    except AttributeError:
        return
    for si, st in enumerate(stages):
        blocks = list(st.blocks)
        for bi, b in enumerate(blocks):
            nxt = blocks[bi + 1] if bi + 1 < len(blocks) else (
                list(stages[si + 1].blocks)[0] if si + 1 < len(stages) else None)
            if nxt is not None and id(nxt.conv1.conv) in _ACAL:
                _NEXT8[id(b)] = id(nxt.conv1.conv)


def _block_fused(self, x):
    """BasicBlockD, fully fused tail: conv1(+norm+act->fp8) -> conv2(+norm) ->
    ONE tail kernel [scSE-apply + residual + lrelu + dual fp16/fp8 emit]."""
    residual = self.skip(x)
    c2key = id(self.conv2.conv)
    es = _ACAL.get(c2key)
    h8 = _cdnr_fused(self.conv1, x, emit_scale=es,
                     emit_inv=None if es is None else 1.0 / _ACALF[c2key])
    out = _cdnr_fused(self.conv2, h8)                    # fp16 NCHW channels-last view
    N, C, D, H, W = out.shape
    M = N * D * H * W
    h_mc = out.permute(0, 2, 3, 4, 1).reshape(M, C)      # storage-order view, free
    res = residual.permute(0, 2, 3, 4, 1)
    res_mc = (res if res.is_contiguous() else res.contiguous()).reshape(M, C)
    gc = gs = None
    if self.apply_se:
        se = self.squeeze_excitation
        gap = h_mc.float().mean(0)                                        # [C]
        z = torch.relu(se.cSE.fc1.weight.data.float().view(-1, C) @ gap
                       + se.cSE.fc1.bias.data.float())
        gc = torch.sigmoid(se.cSE.fc2.weight.data.float().view(C, -1) @ z
                           + se.cSE.fc2.bias.data.float())                # [C]
        gs = torch.sigmoid(
            (h_mc @ se.sSE.conv.weight.data.view(1, C).t().half()).float().squeeze(1)
            + se.sSE.conv.bias.data.float())                              # [M]
    nkey = _NEXT8.get(id(self))
    einv = 1.0 / _ACALF[nkey] if nkey is not None and nkey in _ACALF else None
    y16, y8 = block_tail(h_mc, res_mc, gc, gs,
                         neg_slope=self.nonlin2.negative_slope, emit_inv=einv)
    if y8 is not None:
        _PENDING8[nkey] = Fp8Tensor(y8, _ACAL[nkey], (N, D, H, W, C))
    return y16.reshape(N, D, H, W, C).permute(0, 4, 1, 2, 3)


_TPW = {}   # id(transpconv) -> (wmat fp16 [Cin, 8*Cout], bias fp16 [8*Cout])


def _transp_gemm_raw(tp, x):
    """ConvTranspose3d k2s2 as ONE fp16 GEMM, output left in [Min, 8*Co] layout — the
    pixel-shuffle is decoded in quant_cat_shuffle's addressing, never materialized. Bias
    folds into the GEMM output (NOT norm-absorbed: it enters the NEXT conv's input, where
    zero-padding makes its contribution border-dependent — verified the hard way)."""
    k = id(tp)
    if k not in _TPW:
        w = tp.weight.data                                   # [Cin, Cout, 2,2,2]
        Cin, Co = w.shape[0], w.shape[1]
        wm = w.permute(0, 2, 3, 4, 1).reshape(Cin, 8 * Co).half().contiguous()
        b8 = tp.bias.data.half().repeat(8) if tp.bias is not None else None
        _TPW[k] = (wm, b8, Co)
    wm, b8, Co = _TPW[k]
    N, Cin, Di, Hi, Wi = x.shape
    xm = x.permute(0, 2, 3, 4, 1)
    xm = (xm if xm.is_contiguous() else xm.contiguous()).reshape(-1, Cin).half()
    yg = xm @ wm
    if b8 is not None:
        yg = yg + b8
    return yg, Co, Di, Hi, Wi


def _decoder_fused(self, skips):
    """Decoder with fused cat+quantize: cat(up, skip) goes STRAIGHT to fp8 (never a fp16
    materialization), feeding the stage's CDNR conv."""
    lres = skips[-1]
    for s in range(len(self.stages)):
        tp = self.transpconvs[s]
        sk = skips[-(s + 2)]
        cdnr = self.stages[s].convs[0]
        key = id(cdnr.conv)
        if key in _ACALF:
            yg, CA, Di, Hi, Wi = _transp_gemm_raw(tp, lres)
            N, CB = sk.shape[0], sk.shape[1]
            v = sk.permute(0, 2, 3, 4, 1)
            v = (v if v.is_contiguous() else v.contiguous()).reshape(-1, CB).half()
            y8 = quant_cat_shuffle(yg, v, Di, Hi, Wi, CA, CB, 1.0 / _ACALF[key])
            lres = _cdnr_fused(cdnr, Fp8Tensor(y8, _ACAL[key],
                                               (N, 2 * Di, 2 * Hi, 2 * Wi, CA + CB)))
        else:
            lres = _cdnr_fused(cdnr, torch.cat((tp(lres), sk), 1))
    return self.seg_layers[-1](lres)


@contextlib.contextmanager
def fp8_resident_patched(net=None):
    """Fused resident execution. Calibrate first (one forward under fp8_conv3d_patched()),
    then freeze_calibration(); scales come from _ACAL. Pass the net to enable cross-block
    fp8 emission (block tails quantize directly for their successor's conv1)."""
    from dynamic_network_architectures.building_blocks.simple_conv_blocks import ConvDropoutNormReLU
    from dynamic_network_architectures.building_blocks.residual import BasicBlockD
    if net is not None:
        build_successor_map(net)
    import reference as _ref
    o_cdnr, o_blk = ConvDropoutNormReLU.forward, BasicBlockD.forward
    o_dec = _ref.Decoder.forward
    ConvDropoutNormReLU.forward = lambda self, x: _cdnr_fused(self, x)
    BasicBlockD.forward = _block_fused
    _ref.Decoder.forward = _decoder_fused
    try:
        yield
    finally:
        ConvDropoutNormReLU.forward, BasicBlockD.forward = o_cdnr, o_blk
        _ref.Decoder.forward = o_dec
        _PENDING8.clear()


def mark_fp16_modules(net, keep_last_decoder_stages=0):
    """Keep the convs of the last N decoder stages in fp16 (precision-sensitive selection).
    Returns count kept. Decoder stages nearest the seg head hit the output most directly."""
    _FP16_KEEP.clear()
    if keep_last_decoder_stages <= 0:
        return 0
    # find decoder stages ModuleList (task_decoders.surface.stages or decoder.stages)
    dec_stages = None
    for name, mod in net.named_modules():
        if name.endswith("stages") and "decoder" in name.lower() and isinstance(mod, nn.ModuleList):
            dec_stages = mod
    if dec_stages is None:
        return 0
    n = 0
    for stage in list(dec_stages)[-keep_last_decoder_stages:]:
        for m in stage.modules():
            if isinstance(m, nn.Conv3d) and m.kernel_size == (3, 3, 3):
                _FP16_KEEP.add(id(m))
                n += 1
    return n


@contextlib.contextmanager
def fp8_conv3d_patched():
    nn.Conv3d.forward = _fp8_conv3d_forward
    _FP8_STATS["fp8_calls"] = 0
    _FP8_STATS["fallback_calls"] = 0
    try:
        yield _FP8_STATS
    finally:
        nn.Conv3d.forward = _orig_conv3d_fwd


def surface_dice(pred, gt, tol_vox=2, thr=0.5):
    """SurfaceDice@tol between two probability volumes, thresholded to masks.
    Symmetric fraction of surface voxels within `tol` of the other's surface. Uses a cheap
    dilation-band proxy (3D max-pool) for the tolerance neighborhood."""
    pm = (pred > thr).float().reshape(1, 1, *pred.shape[-3:])
    gm = (gt > thr).float().reshape(1, 1, *gt.shape[-3:])
    # surface = mask minus its 1-voxel erosion (min-pool via negated max-pool)
    def surf(m):
        er = -F.max_pool3d(-m, 3, 1, 1)
        return (m - er).clamp(0, 1)
    ps, gs = surf(pm), surf(gm)
    k = 2 * tol_vox + 1
    gs_band = F.max_pool3d(gs, k, 1, tol_vox)
    ps_band = F.max_pool3d(ps, k, 1, tol_vox)
    p_in = (ps * gs_band).sum()
    g_in = (gs * ps_band).sum()
    denom = ps.sum() + gs.sum()
    if denom < 1:
        return 1.0  # both surfaces empty -> trivially agree
    return ((p_in + g_in) / denom).item()


def load_ct_patch(path, patch, device):
    """Load a CT patch as [1,1,P,P,P] f32, z-scored (the net's expected normalization)."""
    P = patch
    if path and path.endswith(".npy"):
        import numpy as np
        vol = torch.from_numpy(np.load(path)).float()
        while vol.dim() < 3:
            vol = vol.unsqueeze(0)
        d, h, w = vol.shape[-3:]
        z, y, x = (d - P) // 2, (h - P) // 2, (w - P) // 2
        patch_t = vol[..., z:z + P, y:y + P, x:x + P].reshape(1, 1, P, P, P)
    else:
        # synthetic scroll-like CT: smoothed layered noise (no real volume provided)
        torch.manual_seed(0)
        base = torch.randn(1, 1, P, P, P, device=device)
        base = F.avg_pool3d(F.pad(base, (2, 2, 2, 2, 2, 2)), 5, 1, 0)
        layers = torch.sin(torch.linspace(0, 20 * 3.1416, P, device=device)).view(1, 1, 1, 1, P)
        patch_t = (base + 0.6 * layers).to(device)
    patch_t = patch_t.to(device)
    m, s = patch_t.mean(), patch_t.std().clamp(min=1e-6)
    return (patch_t - m) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/home/forrest/fenix/models/surface_recto_3dunet/"
                    "checkpoint_inference_ready.pth")
    ap.add_argument("--ct", default=None, help=".npy CT volume (else synthetic)")
    ap.add_argument("--patch", type=int, default=128)
    ap.add_argument("--keep-fp16-decoder", type=int, default=0,
                    help="keep the last N decoder stages' 3x3x3 convs in fp16")
    ap.add_argument("--fine-scale", action="store_true",
                    help="per-out-channel weight + per-token activation fp8 scaling (simulated)")
    args = ap.parse_args()
    dev = "cuda"
    global _FINE_SCALE
    _FINE_SCALE = args.fine_scale

    from reference import build_and_load
    net = build_and_load(args.ckpt).to(dev).eval()
    kept = mark_fp16_modules(net, args.keep_fp16_decoder)
    if kept:
        print(f"precision-sensitive: {kept} decoder convs kept fp16")

    x = load_ct_patch(args.ct, args.patch, dev)
    print(f"input: {tuple(x.shape)}  device={torch.cuda.get_device_name()}")

    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            y_ref = net(x).float().softmax(1)[:, 1]           # fp16 reference surface prob
        with fp8_conv3d_patched() as stats:
            y_fp8 = net(x).float().softmax(1)[:, 1]           # fp8 3x3x3 convs

    corr = torch.corrcoef(torch.stack([y_fp8.flatten(), y_ref.flatten()]))[0, 1].item()
    maxe = (y_fp8 - y_ref).abs().amax().item()
    sd = surface_dice(y_fp8, y_ref, tol_vox=2)
    dice = (2 * ((y_fp8 > 0.5) & (y_ref > 0.5)).sum()
            / ((y_fp8 > 0.5).sum() + (y_ref > 0.5).sum()).clamp(min=1)).item()

    print(f"\nfp8 3x3x3 convs routed: {stats['fp8_calls']}  (fallback: {stats['fallback_calls']})")
    print(f"prob corr (fp8 vs fp16) : {corr:.5f}")
    print(f"prob max|dP|            : {maxe:.4f}")
    print(f"Dice@0.5                : {dice:.4f}")
    print(f"SurfaceDice@2           : {sd:.4f}   [gate >= 0.998 -> "
          f"{'PASS' if sd >= 0.998 else 'FAIL'}]")


if __name__ == "__main__":
    main()
