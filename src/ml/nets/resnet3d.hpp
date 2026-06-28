// ml/nets/resnet3d.hpp — fenix's from-scratch C++ (torch::nn) reimplementation of the
// ink_canonical_2um model: a 3D ResNet-152 encoder + a 3D-FPN decoder with depth-attention
// collapse to a 2D ink logit (ScrollPrize/villa ink-detection "r152_3ddec"). Module/param names
// mirror the Lightning checkpoint (backbone.* / decoder.* / normalization.*) so weights load by
// name. NOTE: this model consumes a *rendered surface segment* (depth layers × 2D tile), so it
// runs after the flatten/render path — reimplemented + validated here, wired to render later.
//
// Verified against resnetall.py + model_resnet3d_3d_decoder.py:
//   backbone: conv1 7×7×7 s(1,2,2) → bn1 → relu → maxpool (1,3,3)s(1,2,2) → layer1..4
//             (Bottleneck ×[3,8,36,3], expansion 4, BN3d, type-B downsample); returns [x1..x4]
//   decoder:  channel_reduce[i] = Conv1³(no bias)+GroupNorm(min32)+ReLU; top-down trilinear
//             upsample + concat skip → ResConvBlock3D (conv-gn-relu-conv-gn + 1³ shortcut, relu);
//             depth_collapse = softmax(attn_conv,dim=2)·x summed over depth → 2D; logit Conv2d→1
//   input:    normalization = BatchNorm3d(1) (with_norm)
// Only compiled under FENIX_ML.
#pragma once

#include "ml/torch_env.hpp"

#include <vector>

namespace fenix::ml::nets {

namespace nn = torch::nn;
namespace Fn = torch::nn::functional;

// ResNet-D-free standard 3D bottleneck (Hara et al.): 1³ → 3³(stride) → 1³(×4) + downsample.
struct Bottleneck3DImpl : nn::Module {
    nn::Conv3d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    nn::BatchNorm3d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    nn::Sequential downsample{nullptr};
    static constexpr int kExpansion = 4;
    Bottleneck3DImpl(int in_planes, int planes, int stride, bool with_down) {
        conv1 = register_module("conv1", nn::Conv3d(nn::Conv3dOptions(in_planes, planes, 1).bias(false)));
        bn1 = register_module("bn1", nn::BatchNorm3d(planes));
        conv2 = register_module("conv2", nn::Conv3d(nn::Conv3dOptions(planes, planes, 3).stride(stride).padding(1).bias(false)));
        bn2 = register_module("bn2", nn::BatchNorm3d(planes));
        conv3 = register_module("conv3", nn::Conv3d(nn::Conv3dOptions(planes, planes * kExpansion, 1).bias(false)));
        bn3 = register_module("bn3", nn::BatchNorm3d(planes * kExpansion));
        if (with_down) {
            downsample = nn::Sequential(
                nn::Conv3d(nn::Conv3dOptions(in_planes, planes * kExpansion, 1).stride(stride).bias(false)),
                nn::BatchNorm3d(planes * kExpansion));
            register_module("downsample", downsample);
        }
    }
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor res = downsample ? downsample->forward(x) : x;
        auto o = torch::relu(bn1(conv1(x)));
        o = torch::relu(bn2(conv2(o)));
        o = bn3(conv3(o));
        return torch::relu(o + res);
    }
};
TORCH_MODULE(Bottleneck3D);

struct ResNet3DImpl : nn::Module {
    nn::Conv3d conv1{nullptr};
    nn::BatchNorm3d bn1{nullptr};
    nn::MaxPool3d maxpool{nullptr};
    nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    int in_planes = 64;
    ResNet3DImpl(const std::vector<int>& layers, int in_ch) {
        conv1 = register_module("conv1", nn::Conv3d(
            nn::Conv3dOptions(in_ch, 64, {7, 7, 7}).stride({1, 2, 2}).padding({3, 3, 3}).bias(false)));
        bn1 = register_module("bn1", nn::BatchNorm3d(64));
        maxpool = register_module("maxpool", nn::MaxPool3d(
            nn::MaxPool3dOptions({1, 3, 3}).stride({1, 2, 2}).padding({0, 1, 1})));
        layer1 = register_module("layer1", make_layer(64, layers[0], 1));
        layer2 = register_module("layer2", make_layer(128, layers[1], 2));
        layer3 = register_module("layer3", make_layer(256, layers[2], 2));
        layer4 = register_module("layer4", make_layer(512, layers[3], 2));
    }
    nn::Sequential make_layer(int planes, int blocks, int stride) {
        nn::Sequential seq;
        const bool down = stride != 1 || in_planes != planes * Bottleneck3DImpl::kExpansion;
        seq->push_back(Bottleneck3D(in_planes, planes, stride, down));
        in_planes = planes * Bottleneck3DImpl::kExpansion;
        for (int i = 1; i < blocks; ++i) seq->push_back(Bottleneck3D(in_planes, planes, 1, false));
        return seq;
    }
    std::vector<torch::Tensor> forward(torch::Tensor x) {
        x = maxpool(torch::relu(bn1(conv1(x))));
        auto x1 = layer1->forward(x);
        auto x2 = layer2->forward(x1);
        auto x3 = layer3->forward(x2);
        auto x4 = layer4->forward(x3);
        return {x1, x2, x3, x4};
    }
};
TORCH_MODULE(ResNet3D);

// Decoder residual conv block: conv-gn-relu-conv-gn + 1³ shortcut, then relu.
struct ResConvBlock3DImpl : nn::Module {
    nn::Conv3d conv1{nullptr}, conv2{nullptr}, shortcut{nullptr};
    nn::GroupNorm gn1{nullptr}, gn2{nullptr};
    bool has_shortcut;
    ResConvBlock3DImpl(int in_ch, int out_ch) : has_shortcut(in_ch != out_ch) {
        conv1 = register_module("conv1", nn::Conv3d(nn::Conv3dOptions(in_ch, out_ch, 3).padding(1).bias(false)));
        gn1 = register_module("gn1", nn::GroupNorm(nn::GroupNormOptions(std::min(32, out_ch), out_ch)));
        conv2 = register_module("conv2", nn::Conv3d(nn::Conv3dOptions(out_ch, out_ch, 3).padding(1).bias(false)));
        gn2 = register_module("gn2", nn::GroupNorm(nn::GroupNormOptions(std::min(32, out_ch), out_ch)));
        if (has_shortcut)
            shortcut = register_module("shortcut", nn::Conv3d(nn::Conv3dOptions(in_ch, out_ch, 1).bias(false)));
    }
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor id = has_shortcut ? shortcut(x) : x;
        x = torch::relu(gn1(conv1(x)));
        x = gn2(conv2(x));
        return torch::relu(x + id);
    }
};
TORCH_MODULE(ResConvBlock3D);

// softmax-over-depth attention pooling: (B,C,D,H,W) -> (B,C,H,W). Name: depth_collapse.attn_conv.
struct DepthCollapseImpl : nn::Module {
    nn::Conv3d attn_conv{nullptr};
    explicit DepthCollapseImpl(int in_ch) {
        attn_conv = register_module("attn_conv", nn::Conv3d(nn::Conv3dOptions(in_ch, 1, 1)));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto a = torch::softmax(attn_conv(x), 2);
        return (x * a).sum(2);
    }
};
TORCH_MODULE(DepthCollapse);

struct Decoder3DImpl : nn::Module {
    nn::ModuleList channel_reduce{nullptr}, decoder_blocks{nullptr};
    std::vector<nn::Sequential> reduce;
    std::vector<ResConvBlock3D> blocks;
    DepthCollapse depth_collapse{nullptr};
    nn::Conv2d logit{nullptr};
    Decoder3DImpl(std::vector<int> enc = {256, 512, 1024, 2048}, std::vector<int> dec = {64, 128, 256, 512}) {
        channel_reduce = nn::ModuleList();
        for (std::size_t i = 0; i < enc.size(); ++i) {
            nn::Sequential s(
                nn::Conv3d(nn::Conv3dOptions(enc[i], dec[i], 1).bias(false)),
                nn::GroupNorm(nn::GroupNormOptions(std::min(32, dec[i]), dec[i])),
                nn::ReLU(nn::ReLUOptions(true)));
            reduce.push_back(s);
            channel_reduce->push_back(s);
        }
        register_module("channel_reduce", channel_reduce);
        decoder_blocks = nn::ModuleList();
        for (int i = static_cast<int>(dec.size()) - 1; i > 0; --i) {
            auto b = ResConvBlock3D(dec[i] + dec[i - 1], dec[i - 1]);
            blocks.push_back(b);
            decoder_blocks->push_back(b);
        }
        register_module("decoder_blocks", decoder_blocks);
        depth_collapse = register_module("depth_collapse", DepthCollapse(dec[0]));
        logit = register_module("logit", nn::Conv2d(nn::Conv2dOptions(dec[0], 1, 1)));
    }
    torch::Tensor forward(std::vector<torch::Tensor> feat) {
        std::vector<torch::Tensor> f;
        for (std::size_t i = 0; i < 4; ++i) f.push_back(reduce[i]->forward(feat[i]));
        auto up = [](torch::Tensor x, const torch::Tensor& ref) {
            return Fn::interpolate(x, Fn::InterpolateFuncOptions()
                .size(std::vector<int64_t>(ref.sizes().begin() + 2, ref.sizes().end()))
                .mode(torch::kTrilinear).align_corners(false));
        };
        torch::Tensor x = f[3];
        x = torch::cat({up(x, f[2]), f[2]}, 1); x = blocks[0](x);
        x = torch::cat({up(x, f[1]), f[1]}, 1); x = blocks[1](x);
        x = torch::cat({up(x, f[0]), f[0]}, 1); x = blocks[2](x);
        x = depth_collapse(x);                          // softmax-over-depth pool -> (B,C,H,W)
        return logit(x);                                // (B,1,H,W)
    }
};
TORCH_MODULE(Decoder3D);

// Full ink model. Names: normalization.*, backbone.*, decoder.*  (== checkpoint).
struct ResNet3DInkImpl : nn::Module {
    nn::BatchNorm3d normalization{nullptr};
    ResNet3D backbone{nullptr};
    Decoder3D decoder{nullptr};
    explicit ResNet3DInkImpl(bool with_norm = true) {
        if (with_norm) normalization = register_module("normalization", nn::BatchNorm3d(1));
        backbone = register_module("backbone", ResNet3D(std::vector<int>{3, 8, 36, 3}, 1));
        decoder = register_module("decoder", Decoder3D());
    }
    // (B,1,D,H,W) -> 2D ink logit (B,1,H/4,W/4).
    torch::Tensor forward(torch::Tensor x) {
        if (normalization) x = normalization(x);
        return decoder(backbone(x));
    }
};
TORCH_MODULE(ResNet3DInk);

}  // namespace fenix::ml::nets
