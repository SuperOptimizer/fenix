"""fp8 e4m3 conv3d op library (fused Triton implicit-GEMM, sm120) — the building blocks
for a full fp8-resident ResEnc-UNet.

Covers the conv shapes the surface/ink net actually uses (src/ml/nets/resenc_unet.hpp):
  - 3x3x3 stride-1 (conv2, non-downsampling conv1)
  - 3x3x3 stride-2 (conv1 at stage boundaries) — strided gather
  - 1x1x1 (scSE fc1/fc2, skip projection) — degenerates to a plain GEMM (tap loop len 1)
InstanceNorm3d + LeakyReLU are fused into the f32 accumulator BEFORE the fp8 requantize
(norm needs f32 stats; doing it post-fp8 would double-quantize). scSE / GAP / sigmoid stay
in f32/fp16 (tiny, not worth fp8).

All convs keep activations fp8 channels-last [N,D,H,W,C] between layers (Fp8Tensor). The
kernel gathers taps in the K-loop (no im2col) and requantizes to a caller-supplied output
scale in the epilogue. Per-tensor symmetric e4m3 scaling (amax/448).

This is the reference/prototype for the eventual C++ Fp8Net adapter; not wired into the
C++ build. Requires torch 2.9+cu128, triton 3.5+, sm120 (or any fp8-capable arch).
"""
import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


def _cfgs():
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
                          num_warps=w, num_stages=s)
            for bm in (64, 128) for bn in (64, 128) for bk in (32, 64)
            for w in (4, 8) for s in (3, 4)]


@triton.autotune(configs=_cfgs(), key=["N", "Cin", "Cout", "D", "H", "W", "KD", "STRIDE"])
@triton.jit
def _conv3d_f8(
    x_ptr, w_ptr, g_ptr, b_ptr, y_ptr,          # x,w fp8; g,b (norm scale/bias) f32; y fp8
    xs_ptr, ws_ptr, inv_so, neg_slope,          # xs[M] per-token act scale; ws[Cout] per-oc wt scale
    N, D, H, W, Cin, Cout, Do, Ho, Wo,
    KD: tl.constexpr, STRIDE: tl.constexpr, DO_NORM: tl.constexpr, DO_ACT: tl.constexpr,
    FINE_A: tl.constexpr, FINE_W: tl.constexpr, OUT_DTYPE: tl.constexpr,  # 0=fp8 1=fp16 2=f32
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # output spatial is Do*Ho*Wo (strided); each program does BLOCK_M out-rows x BLOCK_N cout.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    wo = offs_m % Wo
    t = offs_m // Wo
    ho = t % Ho
    t = t // Ho
    do = t % Do
    n_idx = t // Do
    m_valid = offs_m < (N * Do * Ho * Wo)
    pad = (KD - 1) // 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    NTAP: tl.constexpr = KD * KD * KD
    for tap in range(NTAP):
        kz = tap // (KD * KD)
        ky = (tap % (KD * KD)) // KD
        kx = tap % KD
        zz = do * STRIDE + kz - pad
        yy = ho * STRIDE + ky - pad
        xx = wo * STRIDE + kx - pad
        in_bounds = m_valid & (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        base = ((n_idx * D + zz) * H + yy) * W + xx
        if FINE_A:
            # per-token act scale: every row of this tap's tile gathers ONE input voxel, whose
            # scale differs per row and per tap -> must be applied per-tap, not in the epilogue.
            xs_tap = tl.load(xs_ptr + base, mask=in_bounds, other=0.0)
            tap_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, Cin, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_valid = offs_k < Cin
                a = tl.load(x_ptr + base[:, None] * Cin + offs_k[None, :],
                            mask=in_bounds[:, None] & k_valid[None, :], other=0.0)
                w_off = offs_n[:, None] * (NTAP * Cin) + (tap * Cin + offs_k)[None, :]
                b_ = tl.load(w_ptr + w_off, mask=(offs_n < Cout)[:, None] & k_valid[None, :], other=0.0)
                tap_acc += tl.dot(a, b_.T)
            acc += xs_tap[:, None] * tap_acc
        else:
            for k0 in range(0, Cin, BLOCK_K):
                offs_k = k0 + tl.arange(0, BLOCK_K)
                k_valid = offs_k < Cin
                a = tl.load(x_ptr + base[:, None] * Cin + offs_k[None, :],
                            mask=in_bounds[:, None] & k_valid[None, :], other=0.0)
                w_off = offs_n[:, None] * (NTAP * Cin) + (tap * Cin + offs_k)[None, :]
                b_ = tl.load(w_ptr + w_off, mask=(offs_n < Cout)[:, None] & k_valid[None, :], other=0.0)
                acc += tl.dot(a, b_.T)

    # dequant: activation scale (scalar case; fine case already applied per-tap) x weight scale
    if not FINE_A:
        acc = acc * tl.load(xs_ptr)                  # scalar act scale
    if FINE_W:
        ws = tl.load(ws_ptr + offs_n, mask=offs_n < Cout, other=0.0)
        acc = acc * ws[None, :]                      # per-out-channel weight scale
    else:
        acc = acc * tl.load(ws_ptr)                  # scalar weight scale
    if DO_NORM:
        # InstanceNorm3d: per (n, cout) channel over spatial. Done here in f32 via a second
        # pass would need cross-program reduction; instead the caller precomputes per-channel
        # (scale=gamma/sqrt(var+eps), shift=beta-mean*scale) and passes as g_ptr/b_ptr[Cout].
        g = tl.load(g_ptr + offs_n, mask=offs_n < Cout, other=1.0)
        bb = tl.load(b_ptr + offs_n, mask=offs_n < Cout, other=0.0)
        acc = acc * g[None, :] + bb[None, :]
    if DO_ACT:
        acc = tl.where(acc >= 0, acc, acc * neg_slope)
    yo = offs_m[:, None] * Cout + offs_n[None, :]
    ymask = m_valid[:, None] & (offs_n < Cout)[None, :]
    if OUT_DTYPE == 0:      # fp8 requant (resident chains)
        acc = acc * inv_so
        acc = tl.minimum(tl.maximum(acc, -448.0), 448.0)     # e4m3 max
        tl.store(y_ptr + yo, acc.to(tl.float8e4nv), mask=ymask)
    elif OUT_DTYPE == 1:    # fp16 (net-integration: feeds InstanceNorm, matches autocast ref)
        tl.store(y_ptr + yo, acc.to(tl.float16), mask=ymask)
    else:                   # f32
        tl.store(y_ptr + yo, acc, mask=ymask)


@triton.jit
def _norm_act_quant(
    h_ptr, y_ptr, mean_ptr, rstd_ptr, g_ptr, b_ptr,
    inv_so, neg_slope, M, C,
    ACT: tl.constexpr, EMIT_FP8: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    # fused InstanceNorm(affine) [+LeakyReLU] [+fp8 quantize]: ONE read + ONE write, replacing
    # the torch norm/act/quantize 3-pass glue between fp8 convs. h [M,C] fp16 channels-last,
    # per-channel mean/rstd precomputed (N=1: one var_mean over rows).
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_m < M)[:, None] & (offs_c < C)[None, :]
    off = offs_m[:, None] * C + offs_c[None, :]
    h = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(mean_ptr + offs_c, mask=offs_c < C, other=0.0)
    rs = tl.load(rstd_ptr + offs_c, mask=offs_c < C, other=0.0)
    g = tl.load(g_ptr + offs_c, mask=offs_c < C, other=1.0)
    b = tl.load(b_ptr + offs_c, mask=offs_c < C, other=0.0)
    y = (h - mu[None, :]) * rs[None, :] * g[None, :] + b[None, :]
    if ACT:
        y = tl.where(y >= 0, y, y * neg_slope)
    if EMIT_FP8:
        y = y * inv_so
        y = tl.minimum(tl.maximum(y, -448.0), 448.0)
        tl.store(y_ptr + off, y.to(tl.float8e4nv), mask=mask)
    else:
        tl.store(y_ptr + off, y.to(tl.float16), mask=mask)


def _cfgs_v2():
    # TD >= 4 configs produce WRONG results (Triton 3.5.1/sm120 miscompile — verified: every
    # TD in {4,8} fails corr, every TD in {1,2} exact; see fp8 design note). Keep TD <= 2.
    # Small tiles (BLOCK_M 16-64) matter for the deep stages: at 8^3/4^3/2^3 spatial only a
    # handful of CTAs launch with big tiles — the GPU idles.
    # BLOCK_K=128 configs crashed BOTH GPUs mid-autotune (unspecified launch failure /
    # heap corruption; one fell off the bus) — keep BK <= 64.
    cfgs = []
    for td, th, tw in ((2, 4, 16), (2, 8, 16), (1, 8, 32), (1, 16, 16),
                       (1, 4, 16), (2, 4, 8), (1, 4, 8), (1, 2, 8), (2, 2, 4)):
        for bn in (32, 64):
            for bk in (32, 64):
                for w in (4, 8):
                    for s in (3, 4):
                        if td * th * tw < 16 or (td * th * tw) * bn > 32768:
                            continue
                        cfgs.append(triton.Config(
                            {"TD": td, "TH": th, "TW": tw, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_cfgs_v2(), key=["N", "Cin", "Cout", "D", "H", "W", "KD", "STRIDE"],
                 reset_to_zero=["ps_ptr", "pq_ptr"])
@triton.jit
def _conv3d_f8_v2(
    x_ptr, w_ptr, y_ptr, xs, ws, ps_ptr, pq_ptr,
    N, D, H, W, Cin, Cout, Do, Ho, Wo,
    KD: tl.constexpr, STRIDE: tl.constexpr, OUT_DTYPE: tl.constexpr,
    STATS: tl.constexpr, SCALE_PTR: tl.constexpr, IS_INT: tl.constexpr,
    WS_VEC: tl.constexpr, xzp, wsum_ptr, AFFINE: tl.constexpr,
    ZP_PTR: tl.constexpr,
    bsm_ptr, BSPARSE: tl.constexpr,
    TD: tl.constexpr, TH: tl.constexpr, TW: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # v2: cube-tiled implicit GEMM. Each program computes a (TD,TH,TW) OUTPUT CUBE x BLOCK_N
    # couts. Neighboring output voxels share most of their 27 taps -> the gather hits L2/L1
    # instead of gmem (~10x reuse vs linear-M). Single flat K-loop over (tap,cin) for deep
    # MMA pipelining. Per-tensor scales (xs, ws floats); fp16/f32 out (norm follows).
    BLOCK_M: tl.constexpr = TD * TH * TW
    # NOTE: a 1D-grid pid_n-fastest swizzle (A-tile L2 reuse) hard-crashes in autotune on
    # Triton 3.5.1/sm120 (cudaErrorLaunchFailure) — same fragility family as the TD>=4
    # miscompile. Keeping the plain 2D grid.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    ncw = tl.cdiv(Wo, TW)
    nch = tl.cdiv(Ho, TH)
    cd = pid_m // (ncw * nch)
    ch = (pid_m // ncw) % nch
    cw = pid_m % ncw
    loc = tl.arange(0, BLOCK_M)
    lw = loc % TW
    lh = (loc // TW) % TH
    ld = loc // (TW * TH)
    odf = cd * TD + ld                 # over N*Do (n folded into depth)
    n_i = odf // Do
    od = odf % Do
    oh = ch * TH + lh
    ow = cw * TW + lw
    m_valid = (odf < N * Do) & (oh < Ho) & (ow < Wo)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    pad = (KD - 1) // 2
    NTAP: tl.constexpr = KD * KD * KD
    K = NTAP * Cin

    # int8 lane (sm86 3090s have int8 TCs at 2x fp16 but NO fp8): int32 accumulate,
    # f32 rescale in the epilogue — dtype of x_ptr/w_ptr drives fp8/fp16/int8 loads
    if IS_INT:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        # block sparsity: whole 32-wide K-groups are pruned to exact zero — skip the
        # gather + MMA for chunks whose covered groups are all pruned (no `continue`
        # in Triton: guard the chunk body)
        if BSPARSE:
            keep = tl.load(bsm_ptr + k0 // 32)
            if BLOCK_K == 64:
                keep = keep | tl.load(bsm_ptr + k0 // 32 + 1)
        else:
            keep = 1
        if keep != 0:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            tap = offs_k // Cin
            cin = offs_k % Cin
            kz = tap // (KD * KD)
            ky = (tap % (KD * KD)) // KD
            kx = tap % KD
            zz = od[:, None] * STRIDE + kz[None, :] - pad
            yy = oh[:, None] * STRIDE + ky[None, :] - pad
            xx = ow[:, None] * STRIDE + kx[None, :] - pad
            ib = (m_valid[:, None] & (offs_k < K)[None, :] &
                  (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W))
            addr = (((n_i[:, None] * D + zz) * H + yy) * W + xx) * Cin + cin[None, :]
            if AFFINE:
                # zero-padding must decode to real 0 = quantized code zp in affine coords
                if ZP_PTR:
                    a = tl.load(x_ptr + addr, mask=ib,
                                other=tl.load(xzp).to(tl.int8))
                else:
                    a = tl.load(x_ptr + addr, mask=ib, other=xzp)
            else:
                a = tl.load(x_ptr + addr, mask=ib, other=0.0)
            w_off = offs_n[:, None] * K + offs_k[None, :]
            wmask = (offs_n < Cout)[:, None] & (offs_k < K)[None, :]
            b = tl.load(w_ptr + w_off, mask=wmask, other=0.0)
            acc += tl.dot(a, b.T)

    if IS_INT:
        if AFFINE:
            wsum = tl.load(wsum_ptr + offs_n, mask=offs_n < Cout, other=0)
            # zp is integral by construction; int32 math keeps the correction
            # EXACT (f32 would round above 2^24 — acc reaches ~5.6e7)
            zpv = tl.load(xzp).to(tl.int32) if ZP_PTR else xzp
            acc = acc - zpv * wsum[None, :]
        acc = acc.to(tl.float32)
    if WS_VEC:
        # per-out-channel weight scales [Cout] — int8's uniform step is sized by the
        # tensor amax, which crushes small-magnitude channels (whole-net SD 0.744
        # per-tensor); per-channel restores the fp8-like relative precision.
        wsv = tl.load(ws + offs_n, mask=offs_n < Cout, other=1.0)
        acc = acc * (tl.load(xs) * wsv[None, :])
    elif SCALE_PTR:
        # xs/ws are 1-elem f32 tensors: no host sync anywhere in the step (P1.3);
        # graph-replay-safe AND scale-updatable without recapture.
        acc = acc * (tl.load(xs) * tl.load(ws))
    else:
        acc = acc * (xs * ws)
    ymo = ((n_i * Do + od) * Ho + oh) * Wo + ow
    yo = ymo[:, None] * Cout + offs_n[None, :]
    ymask = m_valid[:, None] & (offs_n < Cout)[None, :]
    if OUT_DTYPE == 1:
        tl.store(y_ptr + yo, acc.to(tl.float16), mask=ymask)
    else:
        tl.store(y_ptr + yo, acc, mask=ymask)
    if STATS:
        # per-CTA column partials for InstanceNorm — kills the separate stats read pass.
        # 64-way row spread avoids same-address atomic serialization.
        srow = pid_m % 64
        cm = tl.sum(acc, 0)          # invalid rows are exact 0 (masked loads) — safe
        cq = tl.sum(acc * acc, 0)
        tl.atomic_add(ps_ptr + srow * Cout + offs_n, cm, mask=offs_n < Cout)
        tl.atomic_add(pq_ptr + srow * Cout + offs_n, cq, mask=offs_n < Cout)


def fp8_conv3d_v2(x, w_fp8, w_scale_f, Cout, k, stride, out_dtype=torch.float16, xs_f=None,
                  want_stats=False, wsum=None, bsmask=None):
    """Cube-tiled per-tensor fp8 conv3d (x: Fp8Tensor), fp16/f32 out. Scales: python
    floats (xs_f avoids a sync — pass it inside CUDA-graph capture) OR 1-elem f32 device
    tensors (SCALE_PTR path — zero host syncs, the training default; both operands must
    be the same kind). want_stats: also return per-channel (mean, rstd) accumulated in
    the conv epilogue (for InstanceNorm) — saves the separate stats read pass."""
    N, D, H, W, Cin = x.shape
    pad = (k - 1) // 2
    Do = (D + 2 * pad - k) // stride + 1
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    M = N * Do * Ho * Wo
    y = torch.empty(M, Cout, dtype=out_dtype, device=x.data.device)
    if isinstance(w_scale_f, torch.Tensor):
        xs = xs_f if xs_f is not None else x.scale
        scale_ptr = True
    else:
        xs = xs_f if xs_f is not None else float(x.scale)
        scale_ptr = False
    if want_stats:
        ps = torch.zeros(64, Cout, dtype=torch.float32, device=x.data.device)
        pq = torch.zeros(64, Cout, dtype=torch.float32, device=x.data.device)
    else:
        ps = pq = y
    grid = lambda meta: (triton.cdiv(N * Do, meta["TD"]) * triton.cdiv(Ho, meta["TH"])
                         * triton.cdiv(Wo, meta["TW"]),
                         triton.cdiv(Cout, meta["BLOCK_N"]))
    _conv3d_f8_v2[grid](
        x.data, w_fp8, y, xs, w_scale_f, ps, pq,
        N, D, H, W, Cin, Cout, Do, Ho, Wo,
        KD=k, STRIDE=stride, OUT_DTYPE=1 if out_dtype == torch.float16 else 2,
        STATS=want_stats, SCALE_PTR=scale_ptr, IS_INT=x.data.dtype == torch.int8,
        WS_VEC=isinstance(w_scale_f, torch.Tensor) and w_scale_f.numel() > 1,
        xzp=x.zp if getattr(x, "zp", None) is not None else 0,
        wsum_ptr=wsum if wsum is not None else y,
        AFFINE=getattr(x, "zp", None) is not None,
        ZP_PTR=isinstance(getattr(x, "zp", None), torch.Tensor),
        bsm_ptr=bsmask if bsmask is not None else y,
        BSPARSE=bsmask is not None,
    )
    yv = y.reshape(N, Do, Ho, Wo, Cout).permute(0, 4, 1, 2, 3)
    if want_stats:
        s = ps.sum(0)
        mean = s / M
        var = pq.sum(0) / M - mean * mean
        return yv, mean, torch.rsqrt(var.clamp(min=0) + 1e-5)
    return yv


def pack_weight_i8(w, per_channel=True, scales=None):
    """w [Co,Ci,k,k,k] float -> (int8 [Co, k^3*Ci] tap-outer cin-inner, scale [Co]|[1]).
    Symmetric round-to-nearest. per_channel=True is REQUIRED for whole-net accuracy:
    per-tensor int8 scored SD@2 0.744 (uniform step sized by the global amax crushes
    small channels); the kernel's WS_VEC epilogue applies the [Co] vector.
    scales: reuse a prior pack's [Co] scales (skips the amax reduce; per-step weight
    drift is tiny and the clamp handles the tail — refresh every K steps)."""
    Co = w.shape[0]
    if per_channel:
        sb = scales if scales is not None else \
            (w.abs().amax(dim=(1, 2, 3, 4)).clamp(min=1e-8) / 127.0).to(torch.float32)
        q = torch.round(w / sb.view(-1, 1, 1, 1, 1)).clamp(-127, 127).to(torch.int8)
    else:
        sb = (w.abs().amax().clamp(min=1e-8) / 127.0).to(torch.float32).reshape(1)
        q = torch.round(w / sb).clamp(-127, 127).to(torch.int8)
    q = (q.reshape(Co, w.shape[1], -1).permute(0, 2, 1)
          .reshape(Co, -1).contiguous())
    return q, sb.contiguous()


def pack_weight_f16(w):
    """fp16 pack, same [Co, k^3*Ci] layout, unit scale — the v2 kernel is dtype-driven."""
    Co = w.shape[0]
    q = (w.half().reshape(Co, w.shape[1], -1).permute(0, 2, 1)
          .reshape(Co, -1).contiguous())
    return q, torch.ones(1, dtype=torch.float32, device=w.device)


def quantize_i8(x_mc, scale):
    return torch.round(x_mc.float() / scale).clamp(-127, 127).to(torch.int8)


def pack_weight_dgrad_i8(w, per_channel=True):
    """int8 mirror of pack_weight_dgrad_fp8: W'[ci, tap, co] = W[co, mirror(tap), ci].
    per_channel=True -> per-ci [Ci] scales (dgrad's OUTPUT channels are ci, so the
    WS_VEC epilogue applies them exactly like the forward's per-co scales)."""
    Co, Ci, k, _, _ = w.shape
    wr = w.flip(2, 3, 4).permute(1, 2, 3, 4, 0).reshape(Ci, k * k * k * Co)
    if per_channel:
        sb = (wr.abs().amax(dim=1).clamp(min=1e-8) / 127.0).to(torch.float32)
        q = torch.round(wr / sb[:, None]).clamp(-127, 127).to(torch.int8).contiguous()
    else:
        sb = (wr.abs().amax().clamp(min=1e-8) / 127.0).to(torch.float32).reshape(1)
        q = torch.round(wr / sb).clamp(-127, 127).to(torch.int8).contiguous()
    return q, sb


def pack_weight_dgrad_fp8(w):
    """Repack w [Co,Ci,k,k,k] for the DGRAD pass: dx = conv3d(dy, W') with W'[ci, tap, co]
    = W[co, mirror(tap), ci] — dgrad of a stride-1 same-pad conv IS a forward conv with
    mirrored taps and swapped channels, so the v2 forward kernel does dgrad unchanged."""
    Co, Ci, k, _, _ = w.shape
    wr = w.flip(2, 3, 4).permute(1, 2, 3, 4, 0).reshape(Ci, k * k * k * Co)  # [Ci, tap*Co]
    sb = (wr.abs().amax().clamp(min=1e-8) / E4M3_MAX).to(torch.float32).reshape(1)
    q = (wr / sb).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, sb


@triton.autotune(configs=[triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": 32},
                                        num_warps=4, num_stages=2)
                          for bm in (64,) for bn in (32, 64)],
                 key=["N", "Cin", "Cout", "D", "H", "W", "KD"])
@triton.jit
def _dgrad_s2_f8(
    dy_ptr, w_ptr, dx_ptr, sab,
    N, D, H, W, Cin, Cout, Do, Ho, Wo,     # D.. = INPUT dims; Do.. = dy dims
    KD: tl.constexpr, SCALE_PTR: tl.constexpr, IS_INT: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # dgrad of a stride-2 same-pad conv (fractionally-strided conv): for input voxel i,
    # dx[i, ci] = sum_{tap, co} dy[(i + pad - tap)/2, co] W[co, tap, ci] where the division
    # must be exact and in-range (parity-masked taps). M = input voxels; weights use the
    # FORWARD layout [Cout, NTAP*Cin] (no repack needed — we index [co, tap, ci] directly).
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)     # over N*D*H*W (input grid)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)     # over Cin
    xw = offs_m % W
    t = offs_m // W
    xh = t % H
    t = t // H
    xd = t % D
    n_i = t // D
    m_valid = offs_m < (N * D * H * W)
    pad = (KD - 1) // 2
    NTAP: tl.constexpr = KD * KD * KD

    if IS_INT:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for tap in range(NTAP):
        kz = tap // (KD * KD)
        ky = (tap % (KD * KD)) // KD
        kx = tap % KD
        oz2 = xd + pad - kz
        oy2 = xh + pad - ky
        ox2 = xw + pad - kx
        oz = oz2 // 2
        oy = oy2 // 2
        ox = ox2 // 2
        ib = (m_valid & (oz2 % 2 == 0) & (oy2 % 2 == 0) & (ox2 % 2 == 0)
              & (oz >= 0) & (oz < Do) & (oy >= 0) & (oy < Ho) & (ox >= 0) & (ox < Wo))
        base = ((n_i * Do + oz) * Ho + oy) * Wo + ox
        for c0 in range(0, Cout, BLOCK_K):
            offs_c = c0 + tl.arange(0, BLOCK_K)
            cv = offs_c < Cout
            a = tl.load(dy_ptr + base[:, None] * Cout + offs_c[None, :],
                        mask=ib[:, None] & cv[None, :], other=0.0)          # [BM, BK] fp8
            # W[co, tap, ci]: rows co = offs_c, col = tap*Cin + ci
            w_off = offs_c[:, None] * (NTAP * Cin) + (tap * Cin + offs_n)[None, :]
            b = tl.load(w_ptr + w_off,
                        mask=cv[:, None] & (offs_n < Cin)[None, :], other=0.0)  # [BK, BN]
            acc += tl.dot(a, b)

    accf = acc.to(tl.float32) if IS_INT else acc
    accf = accf * (tl.load(sab) if SCALE_PTR else sab)
    yo = offs_m[:, None] * Cin + offs_n[None, :]
    tl.store(dx_ptr + yo, accf.to(tl.float16),
             mask=m_valid[:, None] & (offs_n < Cin)[None, :])


def fp8_conv3d_dgrad_s2(dy_mc, w_fp8_fwd, shapes, Cin, Cout, k, sdy_f, sw_f):
    """dy [Mo,Cout] fp8; w in FORWARD pack layout [Cout, k^3*Cin]. shapes=(N,D,H,W,Do,Ho,Wo)
    with D.. the INPUT dims. Scales: both floats or both 1-elem tensors (sync-free).
    Returns dx fp16 [N,Cin,D,H,W] (channels-last view)."""
    N, D, H, W, Do, Ho, Wo = shapes
    M = N * D * H * W
    dx = torch.empty(M, Cin, dtype=torch.float16, device=dy_mc.device)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Cin, meta["BLOCK_N"]))
    _dgrad_s2_f8[grid](dy_mc, w_fp8_fwd, dx, sdy_f * sw_f,
                       N, D, H, W, Cin, Cout, Do, Ho, Wo, KD=k,
                       SCALE_PTR=isinstance(sdy_f, torch.Tensor),
                       IS_INT=dy_mc.dtype == torch.int8)
    return dx.reshape(N, D, H, W, Cin).permute(0, 4, 1, 2, 3)


def _cfgs_wg():
    # Bounded expansion (2026-07-07 evening): profile showed wgrad at 482us/call vs the
    # fwd's 103us — the ultra-conservative post-crash set (4 configs, w4 s2) left real
    # speed on the table. GPU1's bus-falls were root-caused to hardware (Xid 79 with no
    # compute Xid; GPU0 survived every sweep), so widen moderately ON GPU0: bigger M
    # tiles + warps 8 + stages 3. Still no BLOCK >128, still corr-gated after any change.
    cfgs = []
    for bco, bci in ((32, 32), (64, 64), (64, 32)):
        for bm in (64, 128):
            for sp in (8, 32, 128):   # SPLITS autotuned: fixed 8 leaves single CTAs
                for w in (4, 8):      # looping ~4k iterations at shallow-stage M
                    cfgs.append(triton.Config(
                        {"BLOCK_CO": bco, "BLOCK_CI": bci, "BLOCK_MM": bm,
                         "SPLITS": sp},
                        num_warps=w, num_stages=2))
    return cfgs


@triton.autotune(configs=_cfgs_wg(), key=["N", "Cin", "Cout", "Do", "Ho", "Wo", "KD", "STRIDE"],
                 reset_to_zero=["dw_ptr"])
@triton.jit
def _wgrad_f8(
    dy_ptr, x_ptr, dw_ptr, sdysx,
    N, D, H, W, Cin, Cout, Do, Ho, Wo,
    KD: tl.constexpr, STRIDE: tl.constexpr, SCALE_PTR: tl.constexpr,
    SPLITS: tl.constexpr, TRANS: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_CI: tl.constexpr, BLOCK_MM: tl.constexpr,
):
    # WGRAD: dW[co, tap, ci] = sum_m dy[m, co] * x[shift(m, tap), ci].
    # grid: (tap, co-blk x ci-blk, m-split); per-tap GEMM [Co, M] x [M, Ci] with the
    # M-reduction split across CTAs, f32 atomic accumulation into dW.
    tap = tl.program_id(0)
    pid_cc = tl.program_id(1)
    pid_s = tl.program_id(2)
    nci = tl.cdiv(Cin, BLOCK_CI)
    pid_co = pid_cc // nci
    pid_ci = pid_cc % nci
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    offs_ci = pid_ci * BLOCK_CI + tl.arange(0, BLOCK_CI)
    kz = tap // (KD * KD)
    ky = (tap % (KD * KD)) // KD
    kx = tap % KD
    pad = (KD - 1) // 2
    M = N * Do * Ho * Wo
    rows_per = tl.cdiv(M, SPLITS)
    m0 = pid_s * rows_per

    acc = tl.zeros((BLOCK_CO, BLOCK_CI), dtype=tl.float32)
    for m in range(m0, tl.minimum(m0 + rows_per, M), BLOCK_MM):
        offs_m = m + tl.arange(0, BLOCK_MM)
        wo_ = offs_m % Wo
        t = offs_m // Wo
        ho_ = t % Ho
        t = t // Ho
        do_ = t % Do
        n_ = t // Do
        mv = offs_m < M
        zz = do_ * STRIDE + kz - pad
        yy = ho_ * STRIDE + ky - pad
        xx = wo_ * STRIDE + kx - pad
        ib = mv & (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        if TRANS:
            # coalesced [BM, BCO] load + tl.trans: the pre-transposed addressing below
            # is column-strided (uncoalesced) and cost wgrad ~4.3x the fwd's efficiency.
            # tl.trans-in-dot miscompiled/crashed on Triton 3.5.1/sm120 — re-gated per
            # Triton bump (corr check below MUST pass before this path ships).
            a = tl.load(dy_ptr + offs_m[:, None] * Cout + offs_co[None, :],
                        mask=mv[:, None] & (offs_co < Cout)[None, :], other=0.0)
            a_t = tl.trans(a)                                                   # [BCO, BM]
        else:
            a_t = tl.load(dy_ptr + offs_m[None, :] * Cout + offs_co[:, None],
                          mask=mv[None, :] & (offs_co < Cout)[:, None], other=0.0)
        xb = (((n_ * D + zz) * H + yy) * W + xx)
        b = tl.load(x_ptr + xb[:, None] * Cin + offs_ci[None, :],
                    mask=ib[:, None] & (offs_ci < Cin)[None, :], other=0.0)     # [BM, BCI]
        acc += tl.dot(a_t, b)                                                   # [BCO, BCI]

    acc = acc * (tl.load(sdysx) if SCALE_PTR else sdysx)
    NTAP: tl.constexpr = KD * KD * KD
    dwo = offs_co[:, None] * (NTAP * Cin) + tap * Cin + offs_ci[None, :]
    dmask = (offs_co < Cout)[:, None] & (offs_ci < Cin)[None, :]
    tl.atomic_add(dw_ptr + dwo, acc, mask=dmask)


def _cfgs_wg2():
    # BLOCK_CO/BLOCK_N up to 128 + stages 3 (review item 3): the crash family was
    # BLOCK_K=128 + grid swizzle, NOT wide output tiles — BK stays <= 64.
    cfgs = []
    for bco in (32, 64, 128):
        for bn in (32, 64, 128):
            for bk in (32, 64):
                for sp in (8, 32):
                    for w in (4, 8):
                        for s in (2, 3):
                            cfgs.append(triton.Config(
                                {"BLOCK_CO": bco, "BLOCK_N": bn, "BLOCK_K": bk,
                                 "SPLITS": sp}, num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_cfgs_wg2(), key=["N", "Cin", "Cout", "Do", "Ho", "Wo",
                                           "KD", "STRIDE"], reset_to_zero=["dw_ptr"])
@triton.jit
def _wgrad_f8_v2(
    dy_ptr, x_ptr, dw_ptr, sdysx,
    N, D, H, W, Cin, Cout, Do, Ho, Wo,
    KD: tl.constexpr, STRIDE: tl.constexpr, SCALE_PTR: tl.constexpr,
    SPLITS: tl.constexpr, IS_INT: tl.constexpr,
    bsm_ptr, BSPARSE: tl.constexpr,
    BLOCK_CO: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # WGRAD v2: one GEMM dW[Cout, NTAP*Cin] = dy^T [Cout, M] @ im2col(x) [M, NTAP*Cin],
    # M as the K-loop. The B gather is EXACTLY the fwd kernel's input gather (proven);
    # tap/cin decompose ONCE per CTA (v1 redid the spatial decomposition per tap per
    # chunk — that + 32x32-only tiles was the 4.3x efficiency gap, not load coalescing).
    pid_n = tl.program_id(0)
    pid_co = tl.program_id(1)
    pid_s = tl.program_id(2)
    NTAP: tl.constexpr = KD * KD * KD
    if BSPARSE:
        keep = tl.load(bsm_ptr + (pid_n * BLOCK_N) // 32)
        if BLOCK_N >= 64:
            keep = keep | tl.load(bsm_ptr + (pid_n * BLOCK_N) // 32 + 1)
        if BLOCK_N == 128:
            keep = (keep | tl.load(bsm_ptr + (pid_n * BLOCK_N) // 32 + 2)
                    | tl.load(bsm_ptr + (pid_n * BLOCK_N) // 32 + 3))
        if keep == 0:
            return                      # dw stays zeros for pruned groups (exact)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tap = offs_n // Cin
    cin = offs_n % Cin
    kz = tap // (KD * KD)
    ky = (tap % (KD * KD)) // KD
    kx = tap % KD
    nvalid = offs_n < NTAP * Cin
    offs_co = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    cov = offs_co < Cout
    M = N * Do * Ho * Wo
    rows = tl.cdiv(M, SPLITS)
    # round split boundaries to whole output rows: k0 then starts Wo-aligned, and when
    # BLOCK_K divides Wo the per-chunk (n,od,oh) become SCALARS — kills the per-element
    # div/mod + most of the mask ALU on the dominant shallow stages (review item 7)
    rows = tl.cdiv(rows, Wo) * Wo
    hi = tl.minimum((pid_s + 1) * rows, M)
    pad = (KD - 1) // 2
    aligned = Wo % BLOCK_K == 0
    if IS_INT:
        acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.int32)
    else:
        acc = tl.zeros((BLOCK_CO, BLOCK_N), dtype=tl.float32)
    for k0 in range(pid_s * rows, hi, BLOCK_K):
        a = tl.load(dy_ptr + (k0 + tl.arange(0, BLOCK_K))[None, :] * Cout
                    + offs_co[:, None],
                    mask=((k0 + tl.arange(0, BLOCK_K)) < hi)[None, :] & cov[:, None],
                    other=0.0)                                           # [BCO, BK]
        if aligned:
            row0 = k0 // Wo                       # scalar flat (n,od,oh)
            oh_s = row0 % Ho
            od_s = (row0 // Ho) % Do
            n_s = row0 // (Ho * Do)
            zz1 = od_s * STRIDE + kz - pad                               # [BN]
            yy1 = oh_s * STRIDE + ky - pad                               # [BN]
            xx = ((k0 % Wo) + tl.arange(0, BLOCK_K))[:, None] * STRIDE \
                + kx[None, :] - pad                                      # [BK, BN]
            ib = (nvalid[None, :] & (zz1 >= 0)[None, :] & (zz1 < D)[None, :]
                  & (yy1 >= 0)[None, :] & (yy1 < H)[None, :] & (xx >= 0) & (xx < W))
            addr = (((n_s * D + zz1[None, :]) * H + yy1[None, :]) * W + xx) * Cin \
                + cin[None, :]
        else:
            offs_m = k0 + tl.arange(0, BLOCK_K)
            mv = offs_m < hi
            n_ = offs_m // (Do * Ho * Wo)
            r = offs_m % (Do * Ho * Wo)
            od = r // (Ho * Wo)
            oh = (r // Wo) % Ho
            ow = r % Wo
            zz = od[:, None] * STRIDE + kz[None, :] - pad
            yy = oh[:, None] * STRIDE + ky[None, :] - pad
            xx = ow[:, None] * STRIDE + kx[None, :] - pad
            ib = (mv[:, None] & nvalid[None, :] &
                  (zz >= 0) & (zz < D) & (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W))
            addr = (((n_[:, None] * D + zz) * H + yy) * W + xx) * Cin + cin[None, :]
        b = tl.load(x_ptr + addr, mask=ib, other=0.0)                    # [BK, BN]
        acc += tl.dot(a, b)
    accf = acc.to(tl.float32) if IS_INT else acc
    accf = accf * (tl.load(sdysx) if SCALE_PTR else sdysx)
    dwo = offs_co[:, None] * (NTAP * Cin) + offs_n[None, :]
    tl.atomic_add(dw_ptr + dwo, accf, mask=cov[:, None] & nvalid[None, :])


WGRAD_TRANS = False    # flipped by the trans probe / callers once corr-gated on this stack
WGRAD_V2 = True        # GEMM-shaped wgrad (fwd-kernel structure); False = per-tap v1


def fp8_conv3d_wgrad(dy_mc, x_mc, shapes, Cin, Cout, k, stride, sdy_f, sx_f, nmask=None):
    """dy [M,Cout] fp8, x [Mx,Cin] fp8 (both channels-last); shapes=(N,D,H,W,Do,Ho,Wo).
    Returns dW f32 [Cout, Cin, k, k, k]. SPLITS is autotuned per shape."""
    N, D, H, W, Do, Ho, Wo = shapes
    dw = torch.zeros(Cout, k * k * k * Cin, dtype=torch.float32, device=dy_mc.device)
    if WGRAD_V2:
        grid = lambda meta: (triton.cdiv(k * k * k * Cin, meta["BLOCK_N"]),
                             triton.cdiv(Cout, meta["BLOCK_CO"]),
                             meta["SPLITS"])
        _wgrad_f8_v2[grid](dy_mc, x_mc, dw, sdy_f * sx_f,
                           N, D, H, W, Cin, Cout, Do, Ho, Wo,
                           KD=k, STRIDE=stride,
                           SCALE_PTR=isinstance(sdy_f, torch.Tensor),
                           IS_INT=dy_mc.dtype == torch.int8,
                           bsm_ptr=nmask if nmask is not None else dw,
                           BSPARSE=nmask is not None)
    else:
        grid = lambda meta: (k * k * k,
                             triton.cdiv(Cout, meta["BLOCK_CO"]) * triton.cdiv(Cin, meta["BLOCK_CI"]),
                             meta["SPLITS"])
        _wgrad_f8[grid](dy_mc, x_mc, dw, sdy_f * sx_f,
                        N, D, H, W, Cin, Cout, Do, Ho, Wo,
                        KD=k, STRIDE=stride, SCALE_PTR=isinstance(sdy_f, torch.Tensor),
                        TRANS=WGRAD_TRANS)
    return dw.reshape(Cout, k * k * k, Cin).permute(0, 2, 1).reshape(Cout, Cin, k, k, k)


def _tuned_kernels():
    # every autotuned kernel in the library, by stable name (fwd + both bwd)
    return {"conv_v2": _conv3d_f8_v2, "dgrad_s2": _dgrad_s2_f8, "wgrad": _wgrad_f8,
            "wgrad_v2": _wgrad_f8_v2}


def dump_tuned(path):
    """Serialize ALL autotuners' per-shape best configs (JSON). Production loads this
    via load_tuned() to skip the multi-minute first-call autotune sweeps (training pays
    them on fwd, dgrad AND wgrad)."""
    import json
    out = {}
    for name, kern in _tuned_kernels().items():
        out[name] = {repr(key): {**cfg.kwargs, "num_warps": cfg.num_warps,
                                 "num_stages": cfg.num_stages}
                     for key, cfg in kern.cache.items()}
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return sum(len(v) for v in out.values())


def load_tuned(path):
    """Pre-seed every autotuner cache from dump_tuned() output. Accepts the old
    flat (conv-only) format too."""
    import ast
    import json
    with open(path) as f:
        d = json.load(f)
    kerns = _tuned_kernels()
    if "conv_v2" not in d:
        d = {"conv_v2": d}                      # legacy flat (conv-only) format
    n = 0
    for name, entries in d.items():
        kern = kerns.get(name)
        if kern is None:
            continue
        for k, v in entries.items():
            kw = {kk: vv for kk, vv in v.items() if kk not in ("num_warps", "num_stages")}
            kern.cache[ast.literal_eval(k)] = triton.Config(
                kw, num_warps=v["num_warps"], num_stages=v["num_stages"])
            n += 1
    return n


@triton.jit
def _quant_kernel(x_ptr, y_ptr, inv_s, zp, M, C, SCALE_PTR: tl.constexpr,
                  INT8: tl.constexpr, AFF: tl.constexpr, BLOCK: tl.constexpr,
                  ZP_PTR: tl.constexpr = False, OBS: tl.constexpr = False,
                  mn_ptr=None, mx_ptr=None):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M * C
    iv = tl.load(inv_s) if SCALE_PTR else inv_s
    raw = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    if OBS:
        # fused range observation (delayed scaling): min/max of THIS batch,
        # consumed as next step's quant range — zero extra passes
        tl.atomic_min(mn_ptr, tl.min(tl.where(mask, raw, float("inf"))))
        tl.atomic_max(mx_ptr, tl.max(tl.where(mask, raw, float("-inf"))))
    v = raw * iv
    if AFF:
        zpv = tl.load(zp) if ZP_PTR else zp
        v = v + zpv                      # affine: code = round(x/s) + zp
    if INT8:
        v = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))   # round-to-nearest
        v = tl.minimum(tl.maximum(v, -127.0), 127.0)
        tl.store(y_ptr + offs, v.to(tl.int8), mask=mask)
    else:
        v = tl.minimum(tl.maximum(v, -448.0), 448.0)
        tl.store(y_ptr + offs, v.to(tl.float8e4nv), mask=mask)


def quantize_fp8(x_mc, scale, inv_s=None):
    """fp16/f32 [M,C] -> fp8 [M,C], ONE pass. inv_s: python float avoids the .item()
    sync inside CUDA-graph capture; if scale is a device tensor the reciprocal stays
    on-device (SCALE_PTR path, zero syncs)."""
    M, C = x_mc.shape
    y = torch.empty(M, C, dtype=torch.float8_e4m3fn, device=x_mc.device)
    n = M * C
    if inv_s is None:
        inv_s = torch.reciprocal(scale) if isinstance(scale, torch.Tensor) else 1.0 / scale
    _quant_kernel[(triton.cdiv(n, 4096),)](x_mc, y, inv_s, 0, M, C,
                                           SCALE_PTR=isinstance(inv_s, torch.Tensor),
                                           INT8=False, AFF=False, BLOCK=4096)
    return y


def quantize_i8_affine(x_mc):
    """Asymmetric per-tensor int8: post-lrelu activations are heavily skewed (negatives
    all ~x0.01) — symmetric int8 wastes half its codes there (whole-net SD 0.744/0.754
    measured); zero-point quantization puts codes where the data is.
    Returns (q int8 [M,C], scale [1] f32, zp int)."""
    mn, mx = torch.aminmax(x_mc)
    mn = torch.minimum(mn, torch.zeros_like(mn))     # keep real-0 representable
    mx = torch.maximum(mx, torch.zeros_like(mx))
    scale = ((mx - mn).float().clamp(min=1e-12) / 254.0).reshape(1)
    zp = torch.round(-mn.float() / scale) - 127      # code for real 0, in [-127, 127]
    q = (torch.round(x_mc.float() / scale) + zp).clamp(-127, 127).to(torch.int8)
    return q, scale, int(zp.item())


def quantize_i8_dyn(x_mc):
    """SYNC-FREE dynamic affine int8: per-batch scale/zp as DEVICE tensors (no
    .item()). Fresh scales beat any frozen calibration on crop-to-crop range
    variation (measured: dynamic plateaus ~0.06 KD loss where recal-50 static
    sits ~0.14); this variant removes the ~85 ms/step of host syncs that made
    dynamic slow. Returns (q int8 [M,C], scale f32 [1], zp f32 [1] tensor)."""
    mn, mx = torch.aminmax(x_mc)
    mn = torch.minimum(mn, torch.zeros_like(mn))
    mx = torch.maximum(mx, torch.zeros_like(mx))
    scale = ((mx - mn).float().clamp(min=1e-12) / 254.0).reshape(1)
    zp = (torch.round(-mn.float() / scale) - 127.0).reshape(1)
    M, C = x_mc.shape
    q = torch.empty(M, C, dtype=torch.int8, device=x_mc.device)
    inv_s = torch.reciprocal(scale)
    _quant_kernel[(triton.cdiv(M * C, 4096),)](x_mc, q, inv_s, zp, M, C,
                                               SCALE_PTR=True, INT8=True,
                                               AFF=True, BLOCK=4096, ZP_PTR=True)
    return q, scale, zp


def quantize_i8_delayed(x_mc, st, headroom=1.5):
    """DELAYED-SCALING affine int8 (TE-style): quantize with LAST step's observed
    range x headroom while atomically observing THIS batch's range in the same
    pass. Dynamic-quality freshness (1 step stale, vs 50+ for any recal cadence)
    at static-path speed (no aminmax reduce — the measured 78 ms/step premium).
    st: per-layer dict (created on first call via a one-time dynamic bootstrap).
    Returns (q int8 [M,C], scale f32 [1], zp f32 [1] device tensor)."""
    M, C = x_mc.shape
    if "mn" not in st:
        q, scale, zp = quantize_i8_dyn(x_mc)          # bootstrap (one aminmax)
        mn, mx = torch.aminmax(x_mc)
        st["mn"] = mn.float().reshape(1).clone()
        st["mx"] = mx.float().reshape(1).clone()
        return q, scale, zp
    mn, mx = st["mn"], st["mx"]
    pad = (mx - mn) * ((headroom - 1.0) / 2.0)
    mn_e = torch.minimum(mn - pad, torch.zeros_like(mn))   # keep 0 representable
    mx_e = torch.maximum(mx + pad, torch.zeros_like(mx))
    scale = ((mx_e - mn_e).clamp(min=1e-12) / 254.0)
    zp = (torch.round(-mn_e / scale) - 127.0).clamp(-127, 127)
    mn.fill_(float("inf"))
    mx.fill_(float("-inf"))
    q = torch.empty(M, C, dtype=torch.int8, device=x_mc.device)
    inv_s = torch.reciprocal(scale)
    _quant_kernel[(triton.cdiv(M * C, 4096),)](x_mc, q, inv_s, zp, M, C,
                                               SCALE_PTR=True, INT8=True,
                                               AFF=True, BLOCK=4096, ZP_PTR=True,
                                               OBS=True, mn_ptr=mn, mx_ptr=mx)
    return q, scale, zp


def _delayed_sym_scale(x_mc, st, qmax, headroom):
    """Shared delayed-scaling state machine for SYMMETRIC quantizers: returns
    this step's scale (last observed amax x headroom / qmax) and arms st's
    mn/mx accumulators for the fused OBS pass. Bootstraps with one amax reduce."""
    if "mn" not in st:
        amax = (torch.linalg.vector_norm(x_mc, torch.inf).float()
                .clamp(min=1e-12))
        scale = (amax / qmax).reshape(1)
        st["mn"] = torch.full((1,), float("inf"), device=x_mc.device)
        st["mx"] = torch.full((1,), float("-inf"), device=x_mc.device)
    else:
        mn, mx = st["mn"], st["mx"]
        # read BEFORE reset (stream-ordered: maximum() enqueued before fill_)
        amax = torch.maximum(-mn, mx).clamp(min=1e-12)
        scale = (amax * (headroom / qmax)).reshape(1)
        mn.fill_(float("inf"))
        mx.fill_(float("-inf"))
    return scale


def quantize_i8_delayed_sym(x_mc, st, headroom=2.0):
    """DELAYED-SCALING symmetric int8. MEASURED DEAD END for GRADIENTS (dy):
    1500-step KD drifted to 0.095-0.182 (vs 0.047-0.10 with a fresh dy amax) —
    dy ranges shift too fast step-to-step and int8 has no exponent headroom to
    absorb a stale scale (fp8 does: see quantize_fp8_delayed). Kept for
    activation-like tensors only. Returns (q int8 [M,C], scale f32 [1])."""
    M, C = x_mc.shape
    q = torch.empty(M, C, dtype=torch.int8, device=x_mc.device)
    scale = _delayed_sym_scale(x_mc, st, 127.0, headroom)
    _quant_kernel[(triton.cdiv(M * C, 4096),)](x_mc, q, torch.reciprocal(scale),
                                               0, M, C, SCALE_PTR=True,
                                               INT8=True, AFF=False, BLOCK=4096,
                                               OBS=True, mn_ptr=st["mn"],
                                               mx_ptr=st["mx"])
    return q, scale


def quantize_fp8_delayed(x_mc, st, headroom=2.0):
    """DELAYED-SCALING e4m3 quantize. MEASURED DEAD END for the bwd_fp8 lane
    (1500-step KD, identical crop schedule): on dy final loss 0.022 vs 0.011
    fresh; on x8 (wgrad's saved operand) 0.071 — for 2-4 ms of 218. Backward
    operands need fresh scales, period (see quantize_i8_delayed_sym for the
    int8 version of the same lesson). Kept for activation-side experiments.
    Returns (q fp8 [M,C], scale f32 [1] tensor)."""
    M, C = x_mc.shape
    q = torch.empty(M, C, dtype=torch.float8_e4m3fn, device=x_mc.device)
    scale = _delayed_sym_scale(x_mc, st, E4M3_MAX, headroom)
    _quant_kernel[(triton.cdiv(M * C, 4096),)](x_mc, q, torch.reciprocal(scale),
                                               0, M, C, SCALE_PTR=True,
                                               INT8=False, AFF=False, BLOCK=4096,
                                               OBS=True, mn_ptr=st["mn"],
                                               mx_ptr=st["mx"])
    return q, scale


def wsum_i8(w_i8):
    """Per-out-channel int32 sums of the packed int8 weights [Co, K] — the affine
    zero-point epilogue correction term."""
    return w_i8.to(torch.int32).sum(1).contiguous()


def quantize_i8_fused(x_mc, scale):
    """fp16/f32 [M,C] -> int8 [M,C], one fused pass (round-to-nearest, symmetric)."""
    M, C = x_mc.shape
    y = torch.empty(M, C, dtype=torch.int8, device=x_mc.device)
    inv_s = torch.reciprocal(scale) if isinstance(scale, torch.Tensor) else 1.0 / scale
    _quant_kernel[(triton.cdiv(M * C, 4096),)](x_mc, y, inv_s, 0, M, C,
                                               SCALE_PTR=isinstance(inv_s, torch.Tensor),
                                               INT8=True, AFF=False, BLOCK=4096)
    return y


def quantize_i8_static(x_mc, scale, zp):
    """AFFINE int8 with a FROZEN calibrated (scale, zp) — no reduce, no sync: the
    steady-state inference quantizer (dynamic quantize_i8_affine syncs per call)."""
    M, C = x_mc.shape
    y = torch.empty(M, C, dtype=torch.int8, device=x_mc.device)
    inv_s = torch.reciprocal(scale) if isinstance(scale, torch.Tensor) else 1.0 / scale
    _quant_kernel[(triton.cdiv(M * C, 4096),)](x_mc, y, inv_s, zp, M, C,
                                               SCALE_PTR=isinstance(inv_s, torch.Tensor),
                                               INT8=True, AFF=True, BLOCK=4096)
    return y


@triton.jit
def _colstats_kernel(x_ptr, psum_ptr, psq_ptr, M, C, SPLITS,
                     BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    # partial per-channel sum/sumsq over an M-split; f32 accumulate, fp16 read ONCE, no copy
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    rows_per = tl.cdiv(M, SPLITS)
    m0 = pid_s * rows_per
    s = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sq = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for m in range(m0, tl.minimum(m0 + rows_per, M), BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask = (offs_m < M)[:, None] & (offs_c < C)[None, :]
        v = tl.load(x_ptr + offs_m[:, None] * C + offs_c[None, :], mask=mask, other=0.0).to(tl.float32)
        s += tl.sum(v, axis=0)
        sq += tl.sum(v * v, axis=0)
    o = pid_s * C + offs_c
    tl.store(psum_ptr + o, s, mask=offs_c < C)
    tl.store(psq_ptr + o, sq, mask=offs_c < C)


def col_stats(x_mc, eps=1e-5, splits=64):
    """per-channel (mean, rstd) of fp16 [M,C] with f32 accumulation — one data read."""
    M, C = x_mc.shape
    splits = min(splits, max(1, M // 1024))
    ps = torch.empty(splits, C, dtype=torch.float32, device=x_mc.device)
    pq = torch.empty(splits, C, dtype=torch.float32, device=x_mc.device)
    BC = min(64, triton.next_power_of_2(C))
    _colstats_kernel[(splits, triton.cdiv(C, BC))](x_mc, ps, pq, M, C, splits,
                                                   BLOCK_M=64, BLOCK_C=BC)
    s = ps.sum(0)
    sq = pq.sum(0)
    mean = s / M
    var = sq / M - mean * mean
    return mean, torch.rsqrt(var.clamp(min=0) + eps)


def fused_norm_act(h_mc, gamma, beta, act, neg_slope=0.01, eps=1e-5, out_fp8_scale=None,
                   inv_out=None, stats=None):
    """h_mc: conv output [M,C] fp16 channels-last (N=1). InstanceNorm+affine [+LeakyReLU],
    emitting fp16 [M,C] or (out_fp8_scale given) fp8 [M,C] for the next conv. Conv BIAS must
    be omitted upstream: a per-channel constant is absorbed exactly by the norm's mean.
    inv_out: python-float 1/out_fp8_scale (avoids .item() sync for CUDA-graph capture)."""
    M, C = h_mc.shape
    mean, rstd = stats if stats is not None else col_stats(h_mc, eps=eps)
    emit8 = out_fp8_scale is not None
    y = torch.empty(M, C, dtype=torch.float8_e4m3fn if emit8 else torch.float16,
                    device=h_mc.device)
    BM, BC = 128, min(64, triton.next_power_of_2(C))
    grid = (triton.cdiv(M, BM), triton.cdiv(C, BC))
    if emit8 and inv_out is None:
        inv_out = (1.0 / out_fp8_scale).item()
    _norm_act_quant[grid](
        h_mc, y, mean, rstd, gamma, beta,
        inv_out if emit8 else 1.0, neg_slope, M, C,
        ACT=act, EMIT_FP8=emit8, BLOCK_M=BM, BLOCK_C=BC,
    )
    return y


@triton.jit
def _in_stats(x_ptr, s_ptr, q_ptr, S, C, SPLITS,
              BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # per-(instance, channel) partial sum/sumsq with an S-split grid — a (N, C/BC)-only
    # grid serializes 2M rows through ONE CTA at shallow stages (measured +119 ms/step).
    # grid (N, SPLITS, cblocks); f32 atomic accumulation; finalize in torch ([N,C] tiny).
    n = tl.program_id(0)
    sp = tl.program_id(1)
    pc = tl.program_id(2)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    rows = tl.cdiv(S, SPLITS)
    s_acc = tl.zeros((BLOCK_C,), tl.float32)
    q_acc = tl.zeros((BLOCK_C,), tl.float32)
    for s0 in range(sp * rows, tl.minimum((sp + 1) * rows, S), BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        sm = (offs_s < S) & (offs_s < (sp + 1) * rows)
        v = tl.load(x_ptr + (n * S + offs_s)[:, None] * C + offs_c[None, :],
                    mask=sm[:, None] & cm[None, :], other=0.0).to(tl.float32)
        s_acc += tl.sum(v, 0)
        q_acc += tl.sum(v * v, 0)
    tl.atomic_add(s_ptr + n * C + offs_c, s_acc, mask=cm)
    tl.atomic_add(q_ptr + n * C + offs_c, q_acc, mask=cm)


@triton.jit
def _in_norm_act(h_ptr, y_ptr, mean_ptr, rstd_ptr, g_ptr, b_ptr, neg_slope, S, C,
                 inv_qs, qzp, EMIT_I8: tl.constexpr,
                 BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # n-aware InstanceNorm affine + LeakyReLU (neg_slope=1.0 -> no act).
    # EMIT_I8: store the consumer conv's AFFINE-quantized int8 directly (resident
    # port — the consumer skips its quantize pass); else [M,C] fp16.
    n = tl.program_id(0)
    ps = tl.program_id(1)
    pc = tl.program_id(2)
    offs_s = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_s < S)[:, None] & (offs_c < C)[None, :]
    off = (n * S + offs_s)[:, None] * C + offs_c[None, :]
    h = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(mean_ptr + n * C + offs_c, mask=offs_c < C, other=0.0)
    rs = tl.load(rstd_ptr + n * C + offs_c, mask=offs_c < C, other=0.0)
    g = tl.load(g_ptr + offs_c, mask=offs_c < C, other=1.0)
    b = tl.load(b_ptr + offs_c, mask=offs_c < C, other=0.0)
    y = (h - mu[None, :]) * rs[None, :] * g[None, :] + b[None, :]
    y = tl.where(y >= 0, y, y * neg_slope)
    if EMIT_I8:
        # bitwise-identical to quantize_i8_static on the fp16 output: cast to
        # fp16 first (the unfused path quantizes the stored fp16), then the
        # exact affine round-to-nearest-away-from-zero + [-127,127] clamp
        yq = y.to(tl.float16).to(tl.float32)
        v = yq * tl.load(inv_qs) + qzp
        v = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
        v = tl.minimum(tl.maximum(v, -127.0), 127.0)
        tl.store(y_ptr + off, v.to(tl.int8), mask=mask)
    else:
        tl.store(y_ptr + off, y.to(tl.float16), mask=mask)


@triton.jit
def _in_norm_act_train(h_ptr, y_ptr, xh_ptr, sgn_ptr, mean_ptr, rstd_ptr, g_ptr, b_ptr,
                       inv_sxh, neg_slope, S, C,
                       BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # training-mode norm+act, ONE pass, dual emit: y16 (post-act, forward output) +
    # xhat fp8 (FIXED scale — xhat is normalized, |xhat| < 16 always, no amax pass) +
    # exact sign bits (lrelu branch for backward).
    n = tl.program_id(0)
    ps = tl.program_id(1)
    pc = tl.program_id(2)
    offs_s = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_s < S)[:, None] & (offs_c < C)[None, :]
    off = (n * S + offs_s)[:, None] * C + offs_c[None, :]
    h = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    mu = tl.load(mean_ptr + n * C + offs_c, mask=offs_c < C, other=0.0)
    rs = tl.load(rstd_ptr + n * C + offs_c, mask=offs_c < C, other=0.0)
    g = tl.load(g_ptr + offs_c, mask=offs_c < C, other=1.0)
    b = tl.load(b_ptr + offs_c, mask=offs_c < C, other=0.0)
    xhat = (h - mu[None, :]) * rs[None, :]
    pre = xhat * g[None, :] + b[None, :]
    y = tl.where(pre >= 0, pre, pre * neg_slope)
    tl.store(y_ptr + off, y.to(tl.float16), mask=mask)
    xq = tl.minimum(tl.maximum(xhat * inv_sxh, -448.0), 448.0)
    tl.store(xh_ptr + off, xq.to(tl.float8e4nv), mask=mask)
    # sign BITPACK: 8 channel lanes per byte (C % 8 == 0 asserted in the wrapper)
    bits = tl.where(pre >= 0, 1, 0).to(tl.int32) << (offs_c % 8)[None, :]
    packed = tl.sum(tl.reshape(bits, (BLOCK_S, BLOCK_C // 8, 8)), axis=2)
    offs_c8 = pc * (BLOCK_C // 8) + tl.arange(0, BLOCK_C // 8)
    m8 = (offs_s < S)[:, None] & (offs_c8 * 8 < C)[None, :]
    off8 = (n * S + offs_s)[:, None] * (C // 8) + offs_c8[None, :]
    tl.store(sgn_ptr + off8, packed.to(tl.uint8), mask=m8)


XHAT_SCALE = 16.0 / 448.0    # fixed: xhat is unit-variance; |xhat|<16 at any sane M


def in_norm_act_train(h_mc, N, mean, rstd, gamma, beta, neg_slope):
    """Returns (y16 post-act [M,C], xh8 fp8 xhat [M,C], sgn uint8 [M,C]) in one pass."""
    M, C = h_mc.shape
    S = M // N
    assert C % 8 == 0, C
    y = torch.empty(M, C, dtype=torch.float16, device=h_mc.device)
    xh = torch.empty(M, C, dtype=torch.float8_e4m3fn, device=h_mc.device)
    sgn = torch.empty(M, C // 8, dtype=torch.uint8, device=h_mc.device)
    BC = min(64, triton.next_power_of_2(C))
    _in_norm_act_train[(N, triton.cdiv(S, 128), triton.cdiv(C, BC))](
        h_mc, y, xh, sgn, mean, rstd, gamma, beta, 1.0 / XHAT_SCALE, neg_slope,
        S, C, BLOCK_S=128, BLOCK_C=BC)
    return y, xh, sgn


@triton.jit
def _in_bwd_reduce(do_ptr, xh_ptr, sgn_ptr, sxh_ptr, s1_ptr, s2_ptr,
                   neg_slope, S, C, SPLITS,
                   BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # norm+act backward, pass 1 of 2 (S-split grid + f32 atomics — see _in_stats).
    # xhat loaded DIRECTLY (reconstructing (pre-beta)/gamma amplifies fp8 noise by
    # 1/gamma -> dgamma corr 0.46); lrelu branch from the exact saved sign mask.
    # s1[n,c] = sum_S dpre, s2[n,c] = sum_S dpre*xhat (dgamma/dbeta = their n-sums).
    n = tl.program_id(0)
    sp = tl.program_id(1)
    pc = tl.program_id(2)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    sxh = tl.load(sxh_ptr)
    rows = tl.cdiv(S, SPLITS)
    a1 = tl.zeros((BLOCK_C,), tl.float32)
    a2 = tl.zeros((BLOCK_C,), tl.float32)
    for s0 in range(sp * rows, tl.minimum((sp + 1) * rows, S), BLOCK_S):
        offs_s = s0 + tl.arange(0, BLOCK_S)
        mask = ((offs_s < S) & (offs_s < (sp + 1) * rows))[:, None] & cm[None, :]
        off = (n * S + offs_s)[:, None] * C + offs_c[None, :]
        do = tl.load(do_ptr + off, mask=mask, other=0.0).to(tl.float32)
        xhat = tl.load(xh_ptr + off, mask=mask, other=0.0).to(tl.float32) * sxh
        byte8 = tl.load(sgn_ptr + (off // C) * (C // 8) + (offs_c // 8)[None, :],
                        mask=mask, other=255)
        pos = ((byte8 >> (offs_c % 8)[None, :]) & 1) != 0
        dpre = tl.where(pos, do, do * neg_slope)
        a1 += tl.sum(tl.where(mask, dpre, 0.0), 0)
        a2 += tl.sum(tl.where(mask, dpre * xhat, 0.0), 0)
    tl.atomic_add(s1_ptr + n * C + offs_c, a1, mask=cm)
    tl.atomic_add(s2_ptr + n * C + offs_c, a2, mask=cm)


@triton.jit
def _in_bwd_apply(do_ptr, xh_ptr, sgn_ptr, sxh_ptr, g_ptr, rstd_ptr, s1_ptr,
                  s2_ptr, dh_ptr, neg_slope, S, C,
                  BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # pass 2: dh = gamma*rstd*(dpre - s1/S - xhat*s2/S), fp16 [M,C] out.
    n = tl.program_id(0)
    ps = tl.program_id(1)
    pc = tl.program_id(2)
    offs_s = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    mask = (offs_s < S)[:, None] & cm[None, :]
    off = (n * S + offs_s)[:, None] * C + offs_c[None, :]
    sxh = tl.load(sxh_ptr)
    g = tl.load(g_ptr + offs_c, mask=cm, other=1.0)
    rs = tl.load(rstd_ptr + n * C + offs_c, mask=cm, other=0.0)
    s1 = tl.load(s1_ptr + n * C + offs_c, mask=cm, other=0.0)
    s2 = tl.load(s2_ptr + n * C + offs_c, mask=cm, other=0.0)
    do = tl.load(do_ptr + off, mask=mask, other=0.0).to(tl.float32)
    xhat = tl.load(xh_ptr + off, mask=mask, other=0.0).to(tl.float32) * sxh
    byte8 = tl.load(sgn_ptr + (off // C) * (C // 8) + (offs_c // 8)[None, :],
                    mask=mask, other=255)
    pos = ((byte8 >> (offs_c % 8)[None, :]) & 1) != 0
    dpre = tl.where(pos, do, do * neg_slope)
    dh = g[None, :] * rs[None, :] * (dpre - (s1 / S)[None, :] - xhat * (s2 / S)[None, :])
    tl.store(dh_ptr + off, dh.to(tl.float16), mask=mask)


def in_stats(h_mc, N, eps=1e-5):
    M, C = h_mc.shape
    S = M // N
    s = torch.zeros(N, C, dtype=torch.float32, device=h_mc.device)
    q = torch.zeros(N, C, dtype=torch.float32, device=h_mc.device)
    BC = min(64, triton.next_power_of_2(C))
    splits = max(1, min(128, S // 2048))
    _in_stats[(N, splits, triton.cdiv(C, BC))](h_mc, s, q, S, C, splits,
                                               BLOCK_S=128, BLOCK_C=BC)
    mean = s / S
    var = (q / S - mean * mean).clamp(min=0)
    return mean, torch.rsqrt(var + eps)


def in_norm_act(h_mc, N, mean, rstd, gamma, beta, neg_slope, i8_cal=None):
    """i8_cal=(scale 1-elem tensor, zp int): emit the consumer conv's affine int8
    directly (resident port) — returns int8 [M,C] instead of fp16."""
    M, C = h_mc.shape
    S = M // N
    emit = i8_cal is not None
    y = torch.empty(M, C, dtype=torch.int8 if emit else torch.float16,
                    device=h_mc.device)
    if emit:
        inv_qs = torch.reciprocal(i8_cal[0].float())
        qzp = int(i8_cal[1])
    else:
        inv_qs = torch.zeros(1, device=h_mc.device)
        qzp = 0
    BC = min(64, triton.next_power_of_2(C))
    _in_norm_act[(N, triton.cdiv(S, 128), triton.cdiv(C, BC))](
        h_mc, y, mean, rstd, gamma, beta, neg_slope, S, C, inv_qs, qzp,
        EMIT_I8=emit, BLOCK_S=128, BLOCK_C=BC)
    return y


def in_norm_act_bwd(do_mc, xh, sgn, sxh, gamma, rstd, N, neg_slope):
    """Returns (dh fp16 [M,C], dgamma [C], dbeta [C]). xh: saved xhat, fp8 (big M) or
    fp16 (small M — averaging can't fix noise there and the bytes are negligible);
    sxh: its 1-elem f32 scale (1.0 for fp16). sgn: exact sign mask (uint8 [M,C])."""
    M, C = do_mc.shape
    S = M // N
    s1 = torch.zeros(N, C, dtype=torch.float32, device=do_mc.device)
    s2 = torch.zeros(N, C, dtype=torch.float32, device=do_mc.device)
    BC = min(64, triton.next_power_of_2(C))
    splits = max(1, min(128, S // 2048))
    _in_bwd_reduce[(N, splits, triton.cdiv(C, BC))](do_mc, xh, sgn, sxh, s1, s2,
                                                    neg_slope, S, C, splits,
                                                    BLOCK_S=128, BLOCK_C=BC)
    dh = torch.empty(M, C, dtype=torch.float16, device=do_mc.device)
    _in_bwd_apply[(N, triton.cdiv(S, 128), triton.cdiv(C, BC))](
        do_mc, xh, sgn, sxh, gamma, rstd, s1, s2, dh, neg_slope, S, C,
        BLOCK_S=128, BLOCK_C=BC)
    return dh, s2.sum(0), s1.sum(0)


@triton.jit
def _tail_train_fwd(h_ptr, res_ptr, gc_ptr, gs_ptr, y_ptr, h8_ptr, sgn_ptr,
                    inv_sh, neg_slope, SN, C, HAS_SE: tl.constexpr,
                    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # training BasicBlockD tail, ONE pass: y = lrelu(h*(gc+gs) + res) [scSE] or
    # lrelu(h+res); emits y16 + h fp8 (for the gate grads) + exact sign bits.
    # grid (N, s-blocks, c-blocks); gc [N,C], gs [M]; SN = spatial rows per instance.
    n = tl.program_id(0)
    ps = tl.program_id(1)
    pc = tl.program_id(2)
    offs_s = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_s < SN)[:, None] & (offs_c < C)[None, :]
    off = (n * SN + offs_s)[:, None] * C + offs_c[None, :]
    h = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    if HAS_SE:
        gc = tl.load(gc_ptr + n * C + offs_c, mask=offs_c < C, other=0.0)
        gs = tl.load(gs_ptr + n * SN + offs_s, mask=offs_s < SN, other=0.0).to(tl.float32)
        pre = h * (gc[None, :] + gs[:, None]) + res
    else:
        pre = h + res
    y = tl.where(pre >= 0, pre, pre * neg_slope)
    tl.store(y_ptr + off, y.to(tl.float16), mask=mask)
    if HAS_SE:
        hq = tl.minimum(tl.maximum(h * tl.load(inv_sh), -448.0), 448.0)
        tl.store(h8_ptr + off, hq.to(tl.float8e4nv), mask=mask)
    # sign BITPACK (see _in_norm_act_train)
    bits = tl.where(pre >= 0, 1, 0).to(tl.int32) << (offs_c % 8)[None, :]
    packed = tl.sum(tl.reshape(bits, (BLOCK_S, BLOCK_C // 8, 8)), axis=2)
    offs_c8 = pc * (BLOCK_C // 8) + tl.arange(0, BLOCK_C // 8)
    m8 = (offs_s < SN)[:, None] & (offs_c8 * 8 < C)[None, :]
    off8 = (n * SN + offs_s)[:, None] * (C // 8) + offs_c8[None, :]
    tl.store(sgn_ptr + off8, packed.to(tl.uint8), mask=m8)


@triton.jit
def _tail_train_bwd(dy_ptr, h8_ptr, sgn_ptr, gc_ptr, gs_ptr, sh_ptr,
                    dh_ptr, dres_ptr, dgc_ptr, dgs_ptr,
                    neg_slope, SN, C, HAS_SE: tl.constexpr,
                    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr):
    # grid (N, s-blocks, c-blocks). dpre = dy*lrelu'; dres = dpre; dh = dpre*(gc+gs);
    # dgc[n,c] = col-sum dpre*h (atomic), dgs[m] = row-sum dpre*h (atomic over c-blocks).
    n = tl.program_id(0)
    ps = tl.program_id(1)
    pc = tl.program_id(2)
    offs_s = ps * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_s < SN)[:, None] & (offs_c < C)[None, :]
    off = (n * SN + offs_s)[:, None] * C + offs_c[None, :]
    dy = tl.load(dy_ptr + off, mask=mask, other=0.0).to(tl.float32)
    byte8 = tl.load(sgn_ptr + (off // C) * (C // 8) + (offs_c // 8)[None, :],
                    mask=mask, other=255)
    pos = ((byte8 >> (offs_c % 8)[None, :]) & 1) != 0
    dpre = tl.where(pos, dy, dy * neg_slope)
    tl.store(dres_ptr + off, dpre.to(tl.float16), mask=mask)
    if HAS_SE:
        sh = tl.load(sh_ptr)
        h = tl.load(h8_ptr + off, mask=mask, other=0.0).to(tl.float32) * sh
        gc = tl.load(gc_ptr + n * C + offs_c, mask=offs_c < C, other=0.0)
        gs = tl.load(gs_ptr + n * SN + offs_s, mask=offs_s < SN, other=0.0).to(tl.float32)
        dh = dpre * (gc[None, :] + gs[:, None])
        dph = tl.where(mask, dpre * h, 0.0)
        tl.atomic_add(dgc_ptr + n * C + offs_c, tl.sum(dph, 0), mask=offs_c < C)
        tl.atomic_add(dgs_ptr + n * SN + offs_s, tl.sum(dph, 1), mask=offs_s < SN)
    else:
        dh = dpre
    tl.store(dh_ptr + off, dh.to(tl.float16), mask=mask)


def tail_train_fwd(h_mc, res_mc, gc, gs, N, neg_slope, inv_sh):
    M, C = h_mc.shape
    SN = M // N
    y = torch.empty(M, C, dtype=torch.float16, device=h_mc.device)
    has_se = gc is not None
    h8 = (torch.empty(M, C, dtype=torch.float8_e4m3fn, device=h_mc.device)
          if has_se else y)
    assert C % 8 == 0, C
    sgn = torch.empty(M, C // 8, dtype=torch.uint8, device=h_mc.device)
    BC = min(64, triton.next_power_of_2(C))
    _tail_train_fwd[(N, triton.cdiv(SN, 128), triton.cdiv(C, BC))](
        h_mc, res_mc, gc if has_se else y, gs if has_se else y, y, h8, sgn,
        inv_sh if has_se else y, neg_slope, SN, C, HAS_SE=has_se,
        BLOCK_S=128, BLOCK_C=BC)
    return y, (h8 if has_se else None), sgn


def tail_train_bwd(dy_mc, h8, sgn, gc, gs, sh, N, neg_slope):
    M, C = dy_mc.shape
    SN = M // N
    has_se = gc is not None
    dh = torch.empty(M, C, dtype=torch.float16, device=dy_mc.device)
    dres = torch.empty(M, C, dtype=torch.float16, device=dy_mc.device)
    dgc = torch.zeros(N, C, dtype=torch.float32, device=dy_mc.device) if has_se else None
    dgs = torch.zeros(M, dtype=torch.float32, device=dy_mc.device) if has_se else None
    BC = min(64, triton.next_power_of_2(C))
    _tail_train_bwd[(N, triton.cdiv(SN, 128), triton.cdiv(C, BC))](
        dy_mc, h8 if has_se else dh, sgn, gc if has_se else dh,
        gs if has_se else dh, sh if has_se else dh,
        dh, dres, dgc if has_se else dh, dgs if has_se else dh,
        neg_slope, SN, C, HAS_SE=has_se, BLOCK_S=128, BLOCK_C=BC)
    return dh, dres, dgc, dgs


@triton.jit
def _block_tail(h_ptr, res_ptr, gc_ptr, gs_ptr, y16_ptr, y8_ptr,
                inv_so, neg_slope, M, C,
                SE: tl.constexpr, EMIT8: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    # BasicBlock tail in ONE pass: h*(gate_c[c]+gate_s[m]) [scSE] + residual -> LeakyReLU ->
    # dual emit: fp16 (next block's residual) + fp8 (next conv1 input).
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_m < M)[:, None] & (offs_c < C)[None, :]
    off = offs_m[:, None] * C + offs_c[None, :]
    v = tl.load(h_ptr + off, mask=mask, other=0.0).to(tl.float32)
    if SE:
        gc = tl.load(gc_ptr + offs_c, mask=offs_c < C, other=0.0)
        gs = tl.load(gs_ptr + offs_m, mask=offs_m < M, other=0.0)
        v = v * (gc[None, :] + gs[:, None])
    r = tl.load(res_ptr + off, mask=mask, other=0.0).to(tl.float32)
    v = v + r
    v = tl.where(v >= 0, v, v * neg_slope)
    tl.store(y16_ptr + off, v.to(tl.float16), mask=mask)
    if EMIT8:
        q = v * inv_so
        q = tl.minimum(tl.maximum(q, -448.0), 448.0)
        tl.store(y8_ptr + off, q.to(tl.float8e4nv), mask=mask)


def block_tail(h_mc, res_mc, gate_c, gate_s, neg_slope=0.01, emit_inv=None):
    """h,res fp16 [M,C]; gate_c f32 [C] | None; gate_s f32 [M] | None (both or neither).
    Returns (y16 [M,C], y8 [M,C]|None)."""
    M, C = h_mc.shape
    se = gate_c is not None
    y16 = torch.empty(M, C, dtype=torch.float16, device=h_mc.device)
    y8 = torch.empty(M, C, dtype=torch.float8_e4m3fn, device=h_mc.device) \
        if emit_inv is not None else None
    dummy = h_mc  # unused ptr placeholders
    BM, BC = 128, min(64, triton.next_power_of_2(C))
    _block_tail[(triton.cdiv(M, BM), triton.cdiv(C, BC))](
        h_mc, res_mc, gate_c if se else dummy, gate_s if se else dummy,
        y16, y8 if y8 is not None else y16,
        emit_inv if emit_inv is not None else 1.0, neg_slope, M, C,
        SE=se, EMIT8=y8 is not None, BLOCK_M=BM, BLOCK_C=BC)
    return y16, y8


@triton.jit
def _quant_cat2(a_ptr, b_ptr, y_ptr, inv_s, M, CA, CB, BLOCK_M: tl.constexpr,
                BLOCK_C: tl.constexpr):
    # fused cat(channel-dim)+quantize: y_fp8[M, CA+CB] from two fp16 [M,CA],[M,CB] —
    # the decoder's cat(upsampled, skip) never materializes in fp16.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    C = CA + CB
    in_a = offs_c < CA
    mask = (offs_m < M)[:, None] & (offs_c < C)[None, :]
    a_off = offs_m[:, None] * CA + offs_c[None, :]
    b_off = offs_m[:, None] * CB + (offs_c - CA)[None, :]
    va = tl.load(a_ptr + a_off, mask=mask & in_a[None, :], other=0.0).to(tl.float32)
    vb = tl.load(b_ptr + b_off, mask=mask & (~in_a)[None, :], other=0.0).to(tl.float32)
    v = tl.where(in_a[None, :], va, vb) * inv_s
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(y_ptr + offs_m[:, None] * C + offs_c[None, :], v.to(tl.float8e4nv), mask=mask)


@triton.jit
def _quant_cat_shuf(yg_ptr, sk_ptr, y_ptr, inv_s, Di, Hi, Wi, CA, CB,
                    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    # decoder head fusion: A-side reads the transpconv GEMM output [Min, 8*CA] with the
    # k2s2 pixel-shuffle DECODED IN THE ADDRESS (no interleave copy ever materializes);
    # B-side is the skip; output is the fp8 cat feeding the stage conv.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    Do = 2 * Di
    Ho = 2 * Hi
    Wo = 2 * Wi
    M = Do * Ho * Wo
    C = CA + CB
    z = offs_m // (Ho * Wo)
    rem = offs_m % (Ho * Wo)
    yy = rem // Wo
    xx = rem % Wo
    p = (z % 2) * 4 + (yy % 2) * 2 + (xx % 2)
    src = ((z // 2) * Hi + (yy // 2)) * Wi + (xx // 2)
    in_a = offs_c < CA
    mask = (offs_m < M)[:, None] & (offs_c < C)[None, :]
    a_off = src[:, None] * (8 * CA) + (p[:, None] * CA + offs_c[None, :])
    b_off = offs_m[:, None] * CB + (offs_c - CA)[None, :]
    va = tl.load(yg_ptr + a_off, mask=mask & in_a[None, :], other=0.0).to(tl.float32)
    vb = tl.load(sk_ptr + b_off, mask=mask & (~in_a)[None, :], other=0.0).to(tl.float32)
    v = tl.where(in_a[None, :], va, vb) * inv_s
    v = tl.minimum(tl.maximum(v, -448.0), 448.0)
    tl.store(y_ptr + offs_m[:, None] * C + offs_c[None, :], v.to(tl.float8e4nv), mask=mask)


def quant_cat_shuffle(yg, sk_mc, Di, Hi, Wi, CA, CB, inv_s):
    """yg fp16 [Di*Hi*Wi, 8*CA] (transpconv GEMM out); sk fp16 [8*Di*Hi*Wi, CB].
    -> fp8 [8*Di*Hi*Wi, CA+CB] = quantized cat(pixel_shuffle(yg), skip)."""
    M = 8 * Di * Hi * Wi
    y = torch.empty(M, CA + CB, dtype=torch.float8_e4m3fn, device=yg.device)
    BM = 128
    BC = min(128, triton.next_power_of_2(CA + CB))
    _quant_cat_shuf[(triton.cdiv(M, BM), triton.cdiv(CA + CB, BC))](
        yg, sk_mc, y, inv_s, Di, Hi, Wi, CA, CB, BLOCK_M=BM, BLOCK_C=BC)
    return y


def quant_cat2(a_mc, b_mc, inv_s):
    """fp16 [M,CA] ++ [M,CB] -> fp8 [M,CA+CB] in one pass (inv_s: python float)."""
    M, CA = a_mc.shape
    CB = b_mc.shape[1]
    y = torch.empty(M, CA + CB, dtype=torch.float8_e4m3fn, device=a_mc.device)
    BM = 128
    BC = min(128, triton.next_power_of_2(CA + CB))
    _quant_cat2[(triton.cdiv(M, BM), triton.cdiv(CA + CB, BC))](
        a_mc, b_mc, y, inv_s, M, CA, CB, BLOCK_M=BM, BLOCK_C=BC)
    return y


class Fp8Tensor:
    """fp8 e4m3 activation, channels-last [N,D,H,W,C] flattened to [N*D*H*W, C], + scale.
    scale is a 1-elem tensor (per-tensor) or [M] (per-token: one scale per voxel)."""
    def __init__(self, data, scale, shape):
        self.data = data          # [M, C] float8_e4m3fn
        self.scale = scale        # f32 [1] or [M]
        self.shape = shape        # (N, D, H, W, C)

    @staticmethod
    def from_nchw(x, fine=False):
        N, C, D, H, W = x.shape
        xc = x.permute(0, 2, 3, 4, 1).reshape(N * D * H * W, C)   # [M, C]
        if fine:
            scale = (xc.abs().amax(dim=1).clamp(min=1e-8) / E4M3_MAX).to(torch.float32)  # [M]
            d = (xc / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        else:
            scale = (xc.abs().amax().clamp(min=1e-8) / E4M3_MAX).to(torch.float32).reshape(1)
            d = (xc / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return Fp8Tensor(d.contiguous(), scale.contiguous(), (N, D, H, W, C))

    def to_nchw_f32(self):
        N, D, H, W, C = self.shape
        f = self.data.float()
        f = f * (self.scale[:, None] if self.scale.numel() > 1 else self.scale)
        return f.reshape(N, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()


def pack_weight_fp8(w, per_channel=False):
    """w [Cout,Cin,k,k,k] float -> (fp8 [Cout, k^3*Cin] tap-outer cin-inner, scale [1]|[Cout])."""
    Cout, Cin, k, _, _ = w.shape
    if per_channel:
        sb = (w.abs().amax(dim=(1, 2, 3, 4)).clamp(min=1e-8) / E4M3_MAX).to(torch.float32)  # [Cout]
        q = (w / sb.view(-1, 1, 1, 1, 1)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    else:
        sb = (w.abs().amax().clamp(min=1e-8) / E4M3_MAX).to(torch.float32).reshape(1)
        q = (w / sb).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    q = q.reshape(Cout, Cin, k * k * k).permute(0, 2, 1).reshape(Cout, k * k * k * Cin).contiguous()
    return q, sb.contiguous()


def fp8_conv3d(x: Fp8Tensor, w_fp8, w_scale, Cout, k, stride, out_scale,
               norm_scale=None, norm_bias=None, act=False, neg_slope=0.01,
               out_dtype=torch.float16):
    """One fused fp8 conv3d (+optional InstanceNorm affine +optional LeakyReLU).
    out_scale: f32 scalar tensor -> fp8 requant output (resident chains) and returns Fp8Tensor;
    None -> out_dtype (fp16 default — matches the autocast reference and halves the store
    bandwidth vs f32) output [N,Cout,Do,Ho,Wo], feeding the f32 InstanceNorm next.
    norm_scale/norm_bias: precomputed per-Cout-channel affine (see _conv3d_f8 note)."""
    N, D, H, W, Cin = x.shape
    Do = (D + 2 * ((k - 1) // 2) - k) // stride + 1
    Ho = (H + 2 * ((k - 1) // 2) - k) // stride + 1
    Wo = (W + 2 * ((k - 1) // 2) - k) // stride + 1
    M = N * Do * Ho * Wo
    if out_scale is None:
        od = 1 if out_dtype == torch.float16 else 2
        y = torch.empty(M, Cout, dtype=out_dtype, device=x.data.device)
    else:
        od = 0
        y = torch.empty(M, Cout, dtype=torch.float8_e4m3fn, device=x.data.device)
    do_norm = norm_scale is not None
    g = norm_scale if do_norm else x.scale  # dummy ptr when unused
    b = norm_bias if do_norm else x.scale
    fine_a = x.scale.numel() > 1            # per-token activation scale [M]
    fine_w = w_scale.numel() > 1            # per-out-channel weight scale [Cout]
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(Cout, meta["BLOCK_N"]))
    _conv3d_f8[grid](
        x.data, w_fp8, g, b, y,
        x.scale, w_scale, 1.0 if od else (1.0 / out_scale).item(), neg_slope,
        N, D, H, W, Cin, Cout, Do, Ho, Wo,
        KD=k, STRIDE=stride, DO_NORM=do_norm, DO_ACT=act,
        FINE_A=fine_a, FINE_W=fine_w, OUT_DTYPE=od,
    )
    if od:
        # [M,Cout] row-major IS channels_last_3d of [N,Cout,Do,Ho,Wo]: return the view, no copy
        return y.reshape(N, Do, Ho, Wo, Cout).permute(0, 4, 1, 2, 3)
    return Fp8Tensor(y, out_scale.reshape(1), (N, Do, Ho, Wo, Cout))


# ---- channels-last 2x2x2 average pool (the BasicBlockD stride-2 skip) ------------
# AvgPool3d on a channels_last_3d view forces a full NCDHW .contiguous() (measured
# 6.5 ms/patch across the 6 skip modules). These kernels read/write the CL [M, C]
# row layout directly; mean of 8 rows in f32. Backward is an exact gather:
# dx_row = dy_row(parent) / 8.

@triton.jit
def _avgpool2_cl_fwd(x_ptr, y_ptr, N, D, H, W, C,
                     BLOCK_C: tl.constexpr):
    m = tl.program_id(0)
    cb = tl.program_id(1)
    Do, Ho, Wo = D // 2, H // 2, W // 2
    wo = m % Wo
    ho = (m // Wo) % Ho
    do = (m // (Wo * Ho)) % Do
    n = m // (Wo * Ho * Do)
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    for dz in tl.static_range(2):
        for dy in tl.static_range(2):
            for dx in tl.static_range(2):
                row = ((n * D + 2 * do + dz) * H + 2 * ho + dy) * W + 2 * wo + dx
                acc += tl.load(x_ptr + row * C + offs_c, mask=cm,
                               other=0.0).to(tl.float32)
    tl.store(y_ptr + m * C + offs_c, (acc * 0.125).to(tl.float16), mask=cm)


@triton.jit
def _avgpool2_cl_bwd(dy_ptr, dx_ptr, N, D, H, W, C,
                     BLOCK_C: tl.constexpr):
    m = tl.program_id(0)          # INPUT row
    cb = tl.program_id(1)
    Do, Ho, Wo = D // 2, H // 2, W // 2
    x = m % W
    y = (m // W) % H
    z = (m // (W * H)) % D
    n = m // (W * H * D)
    parent = ((n * Do + z // 2) * Ho + y // 2) * Wo + x // 2
    offs_c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = offs_c < C
    g = tl.load(dy_ptr + parent * C + offs_c, mask=cm, other=0.0).to(tl.float32)
    tl.store(dx_ptr + m * C + offs_c, (g * 0.125).to(tl.float16), mask=cm)


class _AvgPool2Cl(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        N, C, D, H, W = x.shape
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0, (D, H, W)
        xm = x.permute(0, 2, 3, 4, 1)
        xm = (xm if xm.is_contiguous() else xm.contiguous()).reshape(-1, C)
        if xm.dtype != torch.float16:
            xm = xm.half()
        Mo = N * (D // 2) * (H // 2) * (W // 2)
        y = torch.empty(Mo, C, dtype=torch.float16, device=x.device)
        bc = min(triton.next_power_of_2(C), 128)
        _avgpool2_cl_fwd[(Mo, triton.cdiv(C, bc))](xm, y, N, D, H, W, C,
                                                   BLOCK_C=bc)
        ctx.meta = (N, C, D, H, W, x.dtype)
        return y.reshape(N, D // 2, H // 2, W // 2, C).permute(0, 4, 1, 2, 3)

    @staticmethod
    def backward(ctx, dy):
        N, C, D, H, W, xdt = ctx.meta
        dym = dy.permute(0, 2, 3, 4, 1)
        dym = (dym if dym.is_contiguous() else dym.contiguous()).reshape(-1, C)
        if dym.dtype != torch.float16:
            dym = dym.half()
        M = N * D * H * W
        dx = torch.empty(M, C, dtype=torch.float16, device=dy.device)
        bc = min(triton.next_power_of_2(C), 128)
        _avgpool2_cl_bwd[(M, triton.cdiv(C, bc))](dym, dx, N, D, H, W, C,
                                                  BLOCK_C=bc)
        out = dx.reshape(N, D, H, W, C).permute(0, 4, 1, 2, 3)
        return out.to(xdt) if xdt != torch.float16 else out


def avgpool2_cl(x):
    """channels-last 2^3 average pool; returns a CL view (permute is free for the
    following 1x1 Fp8Conv3dLayer, which consumes the [M, C] row layout)."""
    return _AvgPool2Cl.apply(x)


@triton.jit
def _quant_cat2_i8(a_ptr, b_ptr, y_ptr, inv_s, zp, M, CA, CB,
                   BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr):
    # int8-AFFINE cat+quantize (decoder cat fusion, resident port): rounding is
    # bitwise-identical to quantize_i8_static on the fp16 inputs.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    C = CA + CB
    in_a = offs_c < CA
    mask = (offs_m < M)[:, None] & (offs_c < C)[None, :]
    a_off = offs_m[:, None] * CA + offs_c[None, :]
    b_off = offs_m[:, None] * CB + (offs_c - CA)[None, :]
    va = tl.load(a_ptr + a_off, mask=mask & in_a[None, :], other=0.0).to(tl.float32)
    vb = tl.load(b_ptr + b_off, mask=mask & (~in_a)[None, :], other=0.0).to(tl.float32)
    v = tl.where(in_a[None, :], va, vb) * tl.load(inv_s) + zp
    v = tl.where(v >= 0, tl.floor(v + 0.5), tl.ceil(v - 0.5))
    v = tl.minimum(tl.maximum(v, -127.0), 127.0)
    tl.store(y_ptr + offs_m[:, None] * C + offs_c[None, :], v.to(tl.int8), mask=mask)


def quant_cat2_i8(a_mc, b_mc, i8_cal):
    """fp16 [M,CA] ++ [M,CB] -> int8 [M,CA+CB] at the consumer conv's frozen
    affine (scale, zp) — one pass, no fp16 cat materialization."""
    M, CA = a_mc.shape
    CB = b_mc.shape[1]
    y = torch.empty(M, CA + CB, dtype=torch.int8, device=a_mc.device)
    inv_s = torch.reciprocal(i8_cal[0].float())
    BM = 128
    BC = min(128, triton.next_power_of_2(CA + CB))
    _quant_cat2_i8[(triton.cdiv(M, BM), triton.cdiv(CA + CB, BC))](
        a_mc, b_mc, y, inv_s, int(i8_cal[1]), M, CA, CB, BLOCK_M=BM, BLOCK_C=BC)
    return y
