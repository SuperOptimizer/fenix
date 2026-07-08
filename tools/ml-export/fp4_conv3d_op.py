"""fp4 (mxfp4/e2m1) conv3d for sm120 consumer Blackwell — weights in fp4, activations fp8.

Builds on the fp8 v2 cube-tiled implicit-GEMM (fp8_conv3d_op.py) but the weight operand is
MX-format fp4: e2m1 elements packed 2-per-uint8 + one e8m0 scale per 32 K-elements, fed to
tl.dot_scaled (native block-scaled MMA on sm120; software-emulated elsewhere). Activations
stay fp8 e4m3 with a per-32-block e8m0 scale of 1.0 experiments-first (per-tensor scale
folded in the epilogue like v2) — mixed fp8xfp4 is the sweet spot: weights are the
bandwidth-heavy operand in conv (re-read per output tile), activations carry the signal
precision.

Packing (host, once per weight): W [Co,Ci,k,k,k] -> K=(tap outer, ci inner) as in v2, then
per-(co, 32-K-block): scale = 2^ceil(log2(amax/4.0)) as e8m0; elements quantized to e2m1
{0,±.5,±1,±1.5,±2,±3,±4,±6} x scale, packed two nibbles/byte (first element low bits).

Status: authored while the GPUs were down (post-crash); numerics validated in
fp4_probe_dot.py first, then the conv. See docs/design/fp8-conv3d-sm120.md.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import triton
import triton.language as tl

E2M1_MAX = 6.0
# e2m1 representable magnitudes
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def pack_weight_fp4(w):
    """w [Co,Ci,k,k,k] f32 -> (packed uint8 [Co, K//2], scales e8m0-uint8 [Co, K//32])
    with K = k^3*Ci (tap outer, ci inner — matches the v2 activation gather order).
    Requires K % 32 == 0 (Ci multiple of 32 handles every net layer except the stem)."""
    Co, Ci, k, _, _ = w.shape
    K = k * k * k * Ci
    assert K % 32 == 0, f"K={K} must be a multiple of 32 for MX blocks"
    wk = w.reshape(Co, Ci, k * k * k).permute(0, 2, 1).reshape(Co, K).float()  # [Co, K]
    blk = wk.reshape(Co, K // 32, 32)
    amax = blk.abs().amax(dim=2).clamp(min=1e-12)                     # [Co, K/32]
    e = torch.ceil(torch.log2(amax / E2M1_MAX))                       # power-of-2 scale
    e = e.clamp(min=-127, max=127)
    scale = torch.pow(2.0, e)                                         # [Co, K/32]
    q = (blk / scale.unsqueeze(2)).clamp(-E2M1_MAX, E2M1_MAX)
    # round to nearest e2m1 level (sign + magnitude)
    lv = _E2M1_LEVELS.to(q.device)
    mag = q.abs().unsqueeze(-1)                                       # [Co,K/32,32,1]
    idx = (mag - lv.view(1, 1, 1, -1)).abs().argmin(dim=-1)           # [Co,K/32,32]
    # e2m1 encoding: bit3 sign, bits2-0 = level index (0..7 -> 0,.5,1,1.5,2,3,4,6)
    code = idx.to(torch.uint8) | ((q < 0).to(torch.uint8) << 3)
    code = code.reshape(Co, K)
    packed = (code[:, 0::2] | (code[:, 1::2] << 4)).contiguous()      # first elem low bits
    e8m0 = (e + 127).to(torch.uint8).reshape(Co, K // 32).contiguous()
    return packed, e8m0


def unpack_weight_fp4(packed, e8m0, Co, K):
    """reference dequant (host validation): -> f32 [Co, K]."""
    lv = _E2M1_LEVELS.to(packed.device)
    lo = packed & 0xF
    hi = packed >> 4
    code = torch.stack([lo, hi], dim=2).reshape(Co, K)
    mag = lv[(code & 7).long()]
    sgn = torch.where((code & 8) > 0, -1.0, 1.0)
    scale = torch.pow(2.0, e8m0.float() - 127.0)                      # [Co, K/32]
    return (sgn * mag).reshape(Co, K // 32, 32) * scale.unsqueeze(2)


def _cfgs_v4():
    cfgs = []
    for td, th, tw in ((2, 4, 16), (2, 8, 16), (1, 8, 32), (1, 4, 16), (1, 2, 8)):
        for bn in (32, 64):
            for bk in (64, 128):          # BK must be a multiple of 32 (MX block) and 64
                for w in (4, 8):          # for the fp4 K-packing (2 elems/byte)
                    for s in (2, 3):
                        cfgs.append(triton.Config(
                            {"TD": td, "TH": th, "TW": tw, "BLOCK_N": bn, "BLOCK_K": bk},
                            num_warps=w, num_stages=s))
    return cfgs


@triton.autotune(configs=_cfgs_v4(), key=["Cin", "Cout", "D", "H", "W", "KD", "STRIDE"])
@triton.jit
def _conv3d_f4w(
    x_ptr, w_ptr, ws_ptr, y_ptr, xs, N, D, H, W, Cin, Cout, Do, Ho, Wo,
    KD: tl.constexpr, STRIDE: tl.constexpr, OUT_DTYPE: tl.constexpr,
    TD: tl.constexpr, TH: tl.constexpr, TW: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # cube-tiled implicit GEMM, fp8 activations x fp4(MX) weights via tl.dot_scaled.
    # A: fp8 e4m3 [BM, BK] gathered as in v2, scale=1 blocks (per-tensor xs in epilogue).
    # B: e2m1 packed [BN, BK//2] + e8m0 scales [BN, BK//32].
    BLOCK_M: tl.constexpr = TD * TH * TW
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
    odf = cd * TD + ld
    n_i = odf // Do
    od = odf % Do
    oh = ch * TH + lh
    ow = cw * TW + lw
    m_valid = (odf < N * Do) & (oh < Ho) & (ow < Wo)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    pad = (KD - 1) // 2
    NTAP: tl.constexpr = KD * KD * KD
    K = NTAP * Cin

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_scale = tl.full((BLOCK_M, BLOCK_K // 32), 127, dtype=tl.uint8)   # 2^0 = 1.0
    for k0 in range(0, K, BLOCK_K):
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
        a = tl.load(x_ptr + addr, mask=ib, other=0.0)                  # fp8 [BM, BK]
        # B packed nibbles: [BN, BK//2] bytes
        offs_kp = (k0 // 2) + tl.arange(0, BLOCK_K // 2)
        bw = tl.load(w_ptr + offs_n[:, None] * (K // 2) + offs_kp[None, :],
                     mask=(offs_n < Cout)[:, None] & (offs_kp < K // 2)[None, :], other=0)
        offs_ks = (k0 // 32) + tl.arange(0, BLOCK_K // 32)
        bs = tl.load(ws_ptr + offs_n[:, None] * (K // 32) + offs_ks[None, :],
                     mask=(offs_n < Cout)[:, None] & (offs_ks < K // 32)[None, :], other=127)
        acc = tl.dot_scaled(a, a_scale, "e4m3",
                            tl.trans(bw), tl.trans(bs), "e2m1",
                            acc=acc, rhs_k_pack=True)
    acc = acc * xs
    ymo = ((n_i * Do + od) * Ho + oh) * Wo + ow
    yo = ymo[:, None] * Cout + offs_n[None, :]
    ymask = m_valid[:, None] & (offs_n < Cout)[None, :]
    if OUT_DTYPE == 1:
        tl.store(y_ptr + yo, acc.to(tl.float16), mask=ymask)
    else:
        tl.store(y_ptr + yo, acc, mask=ymask)


def fp4w_conv3d(x_fp8, x_scale_f, w_packed, w_scales, Cout, k, stride,
                out_dtype=torch.float16):
    """x_fp8: Fp8Tensor-like (data [M,Cin] e4m3, shape (N,D,H,W,Cin)); weights from
    pack_weight_fp4. Returns [N,Cout,Do,Ho,Wo] view (channels-last storage)."""
    N, D, H, W, Cin = x_fp8.shape
    pad = (k - 1) // 2
    Do = (D + 2 * pad - k) // stride + 1
    Ho = (H + 2 * pad - k) // stride + 1
    Wo = (W + 2 * pad - k) // stride + 1
    M = N * Do * Ho * Wo
    y = torch.empty(M, Cout, dtype=out_dtype, device=x_fp8.data.device)
    grid = lambda meta: (triton.cdiv(N * Do, meta["TD"]) * triton.cdiv(Ho, meta["TH"])
                         * triton.cdiv(Wo, meta["TW"]),
                         triton.cdiv(Cout, meta["BLOCK_N"]))
    _conv3d_f4w[grid](
        x_fp8.data, w_packed, w_scales, y, x_scale_f,
        N, D, H, W, Cin, Cout, Do, Ho, Wo,
        KD=k, STRIDE=stride, OUT_DTYPE=1 if out_dtype == torch.float16 else 2,
    )
    return y.reshape(N, Do, Ho, Wo, Cout).permute(0, 4, 1, 2, 3)
