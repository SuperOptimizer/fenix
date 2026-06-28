#!/usr/bin/env python3
"""reference_dino.py — independent PyTorch reference for the dinovol_v2 3D ViT backbone.

Mirrors ScrollPrize/dinovol (dinov2_eva.py + rope.py) exactly: down_projection patch embed,
cls + 4 register tokens, 24 EVA blocks (separate q/v bias, per-block mixed 3D RoPE, SwiGLU+norm
MLP), final LayerNorm. Loads the teacher backbone and emits x_norm_patchtokens. Used to validate
fenix's C++ reimplementation.

  reference_dino.py run <backbone.pt> <in.raw> <D> <H> <W> <out.raw>   # raw f32 patch tokens
  reference_dino.py cmp <a.raw> <b.raw>
"""
import sys, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED = 864; DEPTH = 24; HEADS = 16; PS = 8; NDIM = 3; NREG = 4
HEAD_DIM = EMBED // HEADS                 # 54
MLP_HIDDEN = int(EMBED * (8 / 3))         # 2304


def rotate_half(x):
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class MixedRope(nn.Module):
    def __init__(self):
        super().__init__()
        self.freqs_per_axis = HEAD_DIM // (2 * NDIM)   # 9
        self.num_pairs = HEAD_DIM // 2                  # 27
        self.register_buffer("periods", torch.empty(self.freqs_per_axis))
        self.mix_frequencies = nn.Parameter(torch.empty(HEADS, self.num_pairs, NDIM))

    def coords(self, shape):  # (P,3) in [-1,1], normalize "separate", no aug at inference
        axes = [torch.arange(0.5, s) / s for s in shape]
        c = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, NDIM)
        return 2.0 * c - 1.0

    def embed(self, coords):
        angles = 2 * math.pi * torch.einsum("td,hpd->htp", coords, self.mix_frequencies)
        angles = angles.tile(2)              # (H, P, head_dim)
        return torch.sin(angles), torch.cos(angles)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(EMBED, EMBED * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(EMBED))
        self.register_buffer("k_bias", torch.zeros(EMBED), persistent=False)
        self.v_bias = nn.Parameter(torch.zeros(EMBED))
        self.proj = nn.Linear(EMBED, EMBED)

    def forward(self, x, sin, cos, prefix):
        B, N, C = x.shape
        qkv = F.linear(x, self.qkv.weight, torch.cat((self.q_bias, self.k_bias, self.v_bias)))
        qkv = qkv.reshape(B, N, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)              # (B,H,N,hd)
        qs, ks = q[:, :, prefix:], k[:, :, prefix:]
        q = torch.cat((q[:, :, :prefix], qs * cos + rotate_half(qs) * sin), dim=2)
        k = torch.cat((k[:, :, :prefix], ks * cos + rotate_half(ks) * sin), dim=2)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_g = nn.Linear(EMBED, MLP_HIDDEN)
        self.fc1_x = nn.Linear(EMBED, MLP_HIDDEN)
        self.norm = nn.LayerNorm(MLP_HIDDEN)
        self.fc2 = nn.Linear(MLP_HIDDEN, EMBED)

    def forward(self, x):
        return self.fc2(self.norm(F.silu(self.fc1_g(x)) * self.fc1_x(x)))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(EMBED)
        self.attn = Attn()
        self.rope_embed = MixedRope()
        self.norm2 = nn.LayerNorm(EMBED)
        self.mlp = SwiGLU()

    def forward(self, x, coords, prefix):
        sin, cos = self.rope_embed.embed(coords)
        x = x + self.attn(self.norm1(x), sin, cos, prefix)
        x = x + self.mlp(self.norm2(x))
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_projection = nn.Module()
        self.down_projection.proj = nn.Conv3d(1, EMBED, PS, PS)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, EMBED))
        self.reg_token = nn.Parameter(torch.zeros(1, NREG, EMBED))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, EMBED))
        self.blocks = nn.ModuleList([Block() for _ in range(DEPTH)])
        self.norm = nn.LayerNorm(EMBED)
        self.prefix = 1 + NREG

    def forward(self, x):
        g = self.down_projection.proj(x)               # (B,E,gd,gh,gw)
        gd, gh, gw = g.shape[2:]
        x = g.flatten(2).transpose(1, 2)               # (B,P,E) in d,h,w (ij) order
        B = x.shape[0]
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = torch.cat((x[:, :1], self.reg_token.expand(B, -1, -1), x[:, 1:]), dim=1)
        coords = self.blocks[0].rope_embed.coords((gd, gh, gw)).to(x.dtype)
        for blk in self.blocks:
            x = blk(x, coords, self.prefix)
        x = self.norm(x)
        return x[:, self.prefix:]                       # x_norm_patchtokens (B,P,E)


def load(path):
    net = Backbone().eval()
    raw = torch.load(path, map_location="cpu", weights_only=False)["teacher"]
    sd = {k[len("backbone."):]: v for k, v in raw.items() if k.startswith("backbone.")}
    net.load_state_dict(sd, strict=True)
    print("dino reference: state_dict loaded strict=True")
    return net


def main():
    if sys.argv[1] == "run":
        path, inp, D, H, W, out = sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
        net = load(path)
        x = torch.from_numpy(np.fromfile(inp, dtype=np.float32)).reshape(1, 1, D, H, W)
        with torch.no_grad():
            y = net(x).contiguous().numpy().astype(np.float32)
        y.tofile(out)
        print("run:", out, y.shape)
    elif sys.argv[1] == "cmp":
        a = np.fromfile(sys.argv[2], dtype=np.float32); b = np.fromfile(sys.argv[3], dtype=np.float32)
        n = min(a.size, b.size); a, b = a[:n], b[:n]
        mad = float(np.max(np.abs(a - b))); rel = mad / (float(np.max(np.abs(a))) + 1e-9)
        print(f"cmp: n={n} max_abs_diff={mad:.4e} rel={rel:.4e} corr={float(np.corrcoef(a,b)[0,1]):.6f}")
        print("VERDICT:", "MATCH" if np.corrcoef(a, b)[0, 1] > 0.9999 and rel < 1e-2 else "MISMATCH")


if __name__ == "__main__":
    main()
