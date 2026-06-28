// ml/nets/dinovol.hpp — fenix's from-scratch C++ (torch::nn) reimplementation of the dinovol_v2
// 3D self-supervised ViT backbone (DINOv2/DINOv3-style) from ScrollPrize/dinovol. Module/param
// names mirror the checkpoint's `backbone.*` exactly so the teacher weights load by name.
//
// Architecture (verified against dinov2_eva.py + rope.py):
//   down_projection.proj : Conv3d(1, 864, k=8, s=8)  -> patch grid, flattened (d,h,w) to tokens
//   tokens               : [cls] ++ [4 register] ++ [patches]   (num_prefix = 5)
//   24 × EvaBlock:
//     norm1 (LayerNorm eps 1e-5) -> EvaAttention -> residual
//       attn: qkv Linear(no bias) + separate q_bias/v_bias (k_bias=0); per-head MIXED 3D RoPE
//             on q,k patch tokens (prefix skipped); scaled_dot_product_attention; proj
//     norm2 -> SwiGLU MLP (fc1_g, fc1_x; SiLU(g)*x; LayerNorm; fc2) -> residual
//   final norm; output = x_norm_patchtokens (B, P, 864)
//   mixed RoPE: angles = 2π · einsum(coords[P,3], mix_frequencies[H,P/2,3]); tile(2); rotate_half.
// Only compiled under FENIX_ML.
#pragma once

#include "ml/torch_env.hpp"

#include <cmath>
#include <numbers>
#include <vector>

namespace fenix::ml::nets {

namespace nn = torch::nn;

struct DinoVolConfig {
    int in_channels = 1, embed = 864, depth = 24, heads = 16, patch = 8;
    int num_reg = 4, ndim = 3;
    double mlp_ratio = 8.0 / 3.0;
};

inline torch::Tensor rotate_half(const torch::Tensor& x) {
    auto c = x.chunk(2, -1);
    return torch::cat({-c[1], c[0]}, -1);
}

// Per-block mixed 3D RoPE. periods is a buffer (unused at inference beyond init), the learnable
// mix_frequencies (heads, head_dim/2, 3) define each head's rotation plane.
struct MixedRopeImpl : nn::Module {
    torch::Tensor periods, mix_frequencies;
    int heads, head_dim, ndim;
    MixedRopeImpl(int heads_, int head_dim_, int ndim_) : heads(heads_), head_dim(head_dim_), ndim(ndim_) {
        const int freqs_per_axis = head_dim / (2 * ndim);
        const int num_pairs = head_dim / 2;
        periods = register_buffer("periods", torch::empty({freqs_per_axis}));
        mix_frequencies = register_parameter("mix_frequencies", torch::empty({heads, num_pairs, ndim}));
    }
    // (P,3) normalized coords in [-1,1] for a (gd,gh,gw) patch grid ("separate" normalization).
    torch::Tensor coords(int gd, int gh, int gw, torch::Device dev) {
        auto o = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
        auto ad = (torch::arange(gd, o) + 0.5) / gd;
        auto ah = (torch::arange(gh, o) + 0.5) / gh;
        auto aw = (torch::arange(gw, o) + 0.5) / gw;
        auto mg = torch::meshgrid({ad, ah, aw}, "ij");
        auto c = torch::stack({mg[0], mg[1], mg[2]}, -1).reshape({-1, ndim});
        return 2.0 * c - 1.0;
    }
    std::pair<torch::Tensor, torch::Tensor> embed(const torch::Tensor& coords) {
        auto angles = 2.0 * std::numbers::pi * torch::einsum("td,hpd->htp", {coords, mix_frequencies});
        angles = angles.tile({1, 1, 2});  // (H, P, head_dim)
        return {torch::sin(angles), torch::cos(angles)};
    }
};
TORCH_MODULE(MixedRope);

struct EvaAttnImpl : nn::Module {
    nn::Linear qkv{nullptr}, proj{nullptr};
    torch::Tensor q_bias, v_bias, k_bias;
    int heads, head_dim;
    EvaAttnImpl(int dim, int heads_) : heads(heads_), head_dim(dim / heads_) {
        qkv = register_module("qkv", nn::Linear(nn::LinearOptions(dim, dim * 3).bias(false)));
        q_bias = register_parameter("q_bias", torch::zeros({dim}));
        v_bias = register_parameter("v_bias", torch::zeros({dim}));
        k_bias = torch::zeros({dim});  // EVA: K has no bias (non-persistent buffer upstream)
        proj = register_module("proj", nn::Linear(dim, dim));
    }
    torch::Tensor forward(torch::Tensor x, const torch::Tensor& sin, const torch::Tensor& cos, int prefix) {
        const auto B = x.size(0), N = x.size(1), C = x.size(2);
        auto bias = torch::cat({q_bias, k_bias.to(x.device()), v_bias});
        auto qkvt = torch::linear(x, qkv->weight, bias)
                        .reshape({B, N, 3, heads, head_dim}).permute({2, 0, 3, 1, 4});
        auto q = qkvt[0], k = qkvt[1], v = qkvt[2];  // (B,H,N,hd)
        auto s = sin.unsqueeze(0), c = cos.unsqueeze(0);  // (1,H,P,hd)
        auto qp = q.slice(2, prefix, N), kp = k.slice(2, prefix, N);
        q = torch::cat({q.slice(2, 0, prefix), qp * c + rotate_half(qp) * s}, 2);
        k = torch::cat({k.slice(2, 0, prefix), kp * c + rotate_half(kp) * s}, 2);
        auto o = torch::scaled_dot_product_attention(q, k, v);
        o = o.transpose(1, 2).reshape({B, N, C});
        return proj(o);
    }
};
TORCH_MODULE(EvaAttn);

struct SwiGLUImpl : nn::Module {
    nn::Linear fc1_g{nullptr}, fc1_x{nullptr}, fc2{nullptr};
    nn::LayerNorm norm{nullptr};
    SwiGLUImpl(int dim, int hidden) {
        fc1_g = register_module("fc1_g", nn::Linear(dim, hidden));
        fc1_x = register_module("fc1_x", nn::Linear(dim, hidden));
        norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({hidden})));
        fc2 = register_module("fc2", nn::Linear(hidden, dim));
    }
    torch::Tensor forward(torch::Tensor x) {
        return fc2(norm(torch::silu(fc1_g(x)) * fc1_x(x)));
    }
};
TORCH_MODULE(SwiGLU);

struct EvaBlockImpl : nn::Module {
    nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    EvaAttn attn{nullptr};
    MixedRope rope_embed{nullptr};
    SwiGLU mlp{nullptr};
    EvaBlockImpl(int dim, int heads, int ndim, int hidden) {
        norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim})));
        attn = register_module("attn", EvaAttn(dim, heads));
        rope_embed = register_module("rope_embed", MixedRope(heads, dim / heads, ndim));
        norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim})));
        mlp = register_module("mlp", SwiGLU(dim, hidden));
    }
    torch::Tensor forward(torch::Tensor x, const torch::Tensor& coords, int prefix) {
        auto [sin, cos] = rope_embed->embed(coords);
        x = x + attn(norm1(x), sin, cos, prefix);
        x = x + mlp(norm2(x));
        return x;
    }
};
TORCH_MODULE(EvaBlock);

// Holder so the patch conv is named down_projection.proj.*
struct DownProjImpl : nn::Module {
    nn::Conv3d proj{nullptr};
    DownProjImpl(int in, int embed, int patch) {
        proj = register_module("proj", nn::Conv3d(nn::Conv3dOptions(in, embed, patch).stride(patch)));
    }
    torch::Tensor forward(torch::Tensor x) { return proj(x); }
};
TORCH_MODULE(DownProj);

struct BackboneImpl : nn::Module {
    DownProj down_projection{nullptr};
    torch::Tensor cls_token, reg_token;
    nn::ModuleList blocks{nullptr};
    std::vector<EvaBlock> bl;
    nn::LayerNorm norm{nullptr};
    int prefix;
    explicit BackboneImpl(const DinoVolConfig& c) : prefix(1 + c.num_reg) {
        down_projection = register_module("down_projection", DownProj(c.in_channels, c.embed, c.patch));
        cls_token = register_parameter("cls_token", torch::zeros({1, 1, c.embed}));
        reg_token = register_parameter("reg_token", torch::zeros({1, c.num_reg, c.embed}));
        blocks = nn::ModuleList();
        const int hidden = static_cast<int>(c.embed * c.mlp_ratio);
        for (int i = 0; i < c.depth; ++i) {
            auto b = EvaBlock(c.embed, c.heads, c.ndim, hidden);
            bl.push_back(b);
            blocks->push_back(b);
        }
        register_module("blocks", blocks);
        norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({c.embed})));
    }
    // (B,1,D,H,W) -> patch tokens (B,P,embed).
    torch::Tensor forward(torch::Tensor x) {
        auto g = down_projection(x);  // (B,E,gd,gh,gw)
        const int gd = static_cast<int>(g.size(2)), gh = static_cast<int>(g.size(3)), gw = static_cast<int>(g.size(4));
        x = g.flatten(2).transpose(1, 2);  // (B,P,E), d-major
        const auto B = x.size(0);
        x = torch::cat({cls_token.expand({B, -1, -1}), x}, 1);
        x = torch::cat({x.slice(1, 0, 1), reg_token.expand({B, -1, -1}), x.slice(1, 1)}, 1);
        auto coords = bl[0]->rope_embed->coords(gd, gh, gw, x.device());
        for (auto& b : bl) x = b(x, coords, prefix);
        x = norm(x);
        return x.slice(1, prefix);  // drop cls + reg -> patch tokens
    }
};
TORCH_MODULE(Backbone);

// Root: names everything under backbone.* (== checkpoint teacher keys).
struct DinoVolImpl : nn::Module {
    Backbone backbone{nullptr};
    explicit DinoVolImpl(const DinoVolConfig& c = {}) {
        backbone = register_module("backbone", Backbone(c));
    }
    torch::Tensor forward(torch::Tensor x) { return backbone(x); }
};
TORCH_MODULE(DinoVol);

}  // namespace fenix::ml::nets
