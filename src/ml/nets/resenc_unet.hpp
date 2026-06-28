// ml/nets/resenc_unet.hpp — fenix's from-scratch C++ (torch::nn) reimplementation of the
// nnU-Net-style **Residual-Encoder U-Net with concurrent spatial+channel SE (scSE)** that
// ScrollPrize trains via vesuvius `NetworkFromConfig`. We own the graph; libtorch is just the
// tensor/conv/CUDA library. Module/parameter names mirror the checkpoint EXACTLY
// (shared_encoder.* / task_decoders.surface.*) so trained weights load by name (weights.hpp).
//
// Semantics verified against the upstream source (ScrollPrize/villa, dynamic_network_architectures):
//   ConvDropoutNormReLU : conv(k, pad=(k-1)/2, bias) -> InstanceNorm3d(affine,eps=1e-5) -> LeakyReLU(0.01)
//   BasicBlockD (ResNet-D): res=skip(x); out=conv2(conv1(x)); out=scSE(out); out+=res; LeakyReLU(out)
//     skip = AvgPool3d(stride) [if stride!=1]  ++  1x1 conv+norm [if in!=out]  (Sequential)
//   scSE = cSE(x) + sSE(x);  cSE: GAP->fc1->ReLU->fc2->Sigmoid (rd=make_divisible(C/16,8));
//                            sSE: Sigmoid(conv1x1 C->1)
//   decoder stage s: x=transpconv_s(lres); x=cat(x, skip[-(s+2)]); x=convs(x); (deep-sup OFF)
//
// Only compiled under FENIX_ML.
#pragma once

#include "ml/torch_env.hpp"

#include <cstdint>
#include <vector>

namespace fenix::ml::nets {

namespace nn = torch::nn;

// make_divisible(channels/16, divisor=8, round_limit=0) — timm/nnU-Net scSE reduction.
inline int se_reduced(int channels, double rd_ratio = 1.0 / 16.0, int divisor = 8) {
    const double v = channels * rd_ratio;
    int nv = std::max(divisor, static_cast<int>(v + divisor / 2.0) / divisor * divisor);
    return nv;
}

// conv -> InstanceNorm3d(affine) -> [LeakyReLU(0.01)]. Names: "conv","norm".
struct ConvNormActImpl : nn::Module {
    nn::Conv3d conv{nullptr};
    nn::InstanceNorm3d norm{nullptr};
    bool act{true};
    ConvNormActImpl(int in, int out, int k, int stride, bool nonlin, bool conv_bias = true) : act(nonlin) {
        conv = register_module("conv", nn::Conv3d(nn::Conv3dOptions(in, out, k)
                       .stride(stride).padding((k - 1) / 2).bias(conv_bias)));
        norm = register_module("norm", nn::InstanceNorm3d(nn::InstanceNorm3dOptions(out)
                       .affine(true).track_running_stats(false).eps(1e-5)));
    }
    torch::Tensor forward(torch::Tensor x) {
        x = norm(conv(x));
        if (act) x = torch::leaky_relu(x, 0.01);
        return x;
    }
};
TORCH_MODULE(ConvNormAct);

// Channel SE: x * sigmoid(fc2(relu(fc1(GAP(x))))). Names: "fc1","fc2".
struct CSEImpl : nn::Module {
    nn::Conv3d fc1{nullptr}, fc2{nullptr};
    CSEImpl(int c) {
        const int rd = se_reduced(c);
        fc1 = register_module("fc1", nn::Conv3d(nn::Conv3dOptions(c, rd, 1).bias(true)));
        fc2 = register_module("fc2", nn::Conv3d(nn::Conv3dOptions(rd, c, 1).bias(true)));
    }
    torch::Tensor forward(torch::Tensor x) {
        auto s = x.mean({2, 3, 4}, /*keepdim=*/true);
        s = fc2(torch::relu(fc1(s)));
        return x * torch::sigmoid(s);
    }
};
TORCH_MODULE(CSE);

// Spatial SE: x * sigmoid(conv1x1(x)). Name: "conv".
struct SSEImpl : nn::Module {
    nn::Conv3d conv{nullptr};
    SSEImpl(int c) { conv = register_module("conv", nn::Conv3d(nn::Conv3dOptions(c, 1, 1).bias(true))); }
    torch::Tensor forward(torch::Tensor x) { return x * torch::sigmoid(conv(x)); }
};
TORCH_MODULE(SSE);

// scSE = cSE + sSE. Names: "cSE","sSE".
struct SCSEImpl : nn::Module {
    CSE cse{nullptr}; SSE sse{nullptr};
    SCSEImpl(int c) {
        cse = register_module("cSE", CSE(c));
        sse = register_module("sSE", SSE(c));
    }
    torch::Tensor forward(torch::Tensor x) { return cse(x) + sse(x); }
};
TORCH_MODULE(SCSE);

// Residual block (ResNet-D), optionally with scSE. Names: conv1, conv2, [squeeze_excitation], skip.
struct BasicBlockImpl : nn::Module {
    ConvNormAct conv1{nullptr}, conv2{nullptr};
    SCSE se{nullptr};
    bool has_se{false};
    nn::Sequential skip{nullptr};  // [AvgPool3d?, ConvNormAct(1x1)?]; null = identity
    BasicBlockImpl(int in, int out, int k, int stride, bool squeeze_excitation) : has_se(squeeze_excitation) {
        conv1 = register_module("conv1", ConvNormAct(in, out, k, stride, /*nonlin=*/true));
        conv2 = register_module("conv2", ConvNormAct(out, out, k, 1, /*nonlin=*/false));
        if (has_se) se = register_module("squeeze_excitation", SCSE(out));
        const bool has_stride = stride != 1;
        const bool proj = in != out;
        if (has_stride || proj) {
            skip = nn::Sequential();
            if (has_stride)
                skip->push_back(nn::AvgPool3d(nn::AvgPool3dOptions(stride).stride(stride)));
            if (proj)  // upstream skip projection uses conv_bias=False
                skip->push_back(ConvNormAct(in, out, 1, 1, /*nonlin=*/false, /*conv_bias=*/false));
            register_module("skip", skip);
        }
    }
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor res = skip ? skip->forward(x) : x;
        torch::Tensor out = conv2(conv1(x));
        if (has_se) out = se(out);
        out = out + res;
        return torch::leaky_relu(out, 0.01);
    }
};
TORCH_MODULE(BasicBlock);

// A stack of residual blocks (first carries the stride). Name child: "blocks" (ModuleList).
struct StageImpl : nn::Module {
    nn::ModuleList blocks{nullptr};
    std::vector<BasicBlock> bl;
    StageImpl(int in, int out, int k, int stride, int n, bool se) {
        blocks = nn::ModuleList();
        for (int i = 0; i < n; ++i) {
            auto b = BasicBlock(i == 0 ? in : out, out, k, i == 0 ? stride : 1, se);
            bl.push_back(b);
            blocks->push_back(b);
        }
        register_module("blocks", blocks);
    }
    torch::Tensor forward(torch::Tensor x) {
        for (auto& b : bl) x = b(x);
        return x;
    }
};
TORCH_MODULE(Stage);

// StackedConvBlocks (used for stem + decoder stages). Name child: "convs" (ModuleList).
struct ConvStackImpl : nn::Module {
    nn::ModuleList convs{nullptr};
    std::vector<ConvNormAct> cv;
    ConvStackImpl(int in, int out, int k, int stride, int n) {
        convs = nn::ModuleList();
        for (int i = 0; i < n; ++i) {
            auto c = ConvNormAct(i == 0 ? in : out, out, k, i == 0 ? stride : 1, /*nonlin=*/true);
            cv.push_back(c);
            convs->push_back(c);
        }
        register_module("convs", convs);
    }
    torch::Tensor forward(torch::Tensor x) {
        for (auto& c : cv) x = c(x);
        return x;
    }
};
TORCH_MODULE(ConvStack);

// Residual encoder: stem + stages; returns per-stage outputs (skips). Names: stem, stages.
struct EncoderImpl : nn::Module {
    ConvStack stem{nullptr};
    nn::ModuleList stages{nullptr};
    std::vector<Stage> st;
    EncoderImpl(int in_ch, const std::vector<int>& feats, const std::vector<int>& blocks,
                const std::vector<int>& strides, int k, bool se) {
        stem = register_module("stem", ConvStack(in_ch, feats[0], k, 1, 1));
        stages = nn::ModuleList();
        int prev = feats[0];
        for (std::size_t s = 0; s < feats.size(); ++s) {
            auto stg = Stage(prev, feats[s], k, strides[s], blocks[s], se);
            st.push_back(stg);
            stages->push_back(stg);
            prev = feats[s];
        }
        register_module("stages", stages);
    }
    std::vector<torch::Tensor> forward(torch::Tensor x) {
        x = stem(x);
        std::vector<torch::Tensor> skips;
        for (auto& s : st) { x = s(x); skips.push_back(x); }
        return skips;
    }
};
TORCH_MODULE(Encoder);

// U-Net decoder body: transpconvs + conv stages. Optionally per-stage seg heads (surface,
// deep-sup off → use last). Without seg heads it returns the full-res feature map (ink, where a
// separate task_heads conv produces the logit). Names: transpconvs, stages, [seg_layers].
struct DecoderImpl : nn::Module {
    nn::ModuleList transpconvs{nullptr}, stages{nullptr}, seg_layers{nullptr};
    std::vector<nn::ConvTranspose3d> tp;
    std::vector<ConvStack> dec;
    std::vector<nn::Conv3d> seg;
    bool has_seg{false};
    DecoderImpl(const std::vector<int>& feats, int num_classes, int k, bool with_seg)
        : has_seg(with_seg) {
        transpconvs = nn::ModuleList(); stages = nn::ModuleList();
        const int n = static_cast<int>(feats.size());  // 7 encoder stages -> 6 decoder stages
        if (with_seg) seg_layers = nn::ModuleList();
        for (int s = 0; s < n - 1; ++s) {
            const int below = feats[n - 1 - s];   // deeper feature count (input to transpconv)
            const int skipc = feats[n - 2 - s];   // matching encoder skip channels
            auto t = nn::ConvTranspose3d(nn::ConvTranspose3dOptions(below, skipc, 2).stride(2).bias(true));
            tp.push_back(t); transpconvs->push_back(t);
            auto c = ConvStack(2 * skipc, skipc, k, 1, 1);  // cat(upsampled, skip) -> skipc
            dec.push_back(c); stages->push_back(c);
            if (with_seg) {
                auto sl = nn::Conv3d(nn::Conv3dOptions(skipc, num_classes, 1).bias(true));
                seg.push_back(sl); seg_layers->push_back(sl);
            }
        }
        register_module("transpconvs", transpconvs);
        register_module("stages", stages);
        if (with_seg) register_module("seg_layers", seg_layers);
    }
    torch::Tensor forward(const std::vector<torch::Tensor>& skips) {
        torch::Tensor lres = skips.back();
        const int n = static_cast<int>(skips.size());
        for (std::size_t s = 0; s < tp.size(); ++s) {
            torch::Tensor x = tp[s]->forward(lres);
            x = torch::cat({x, skips[n - 2 - static_cast<int>(s)]}, 1);
            x = dec[s](x);
            lres = x;
        }
        return has_seg ? seg.back()->forward(lres) : lres;  // surface: final seg head; ink: features
    }
};
TORCH_MODULE(Decoder);

// Holder naming a child module under a task name: task_decoders.<task> or task_heads.<task>.
struct TaskWrapImpl : nn::Module {
    TaskWrapImpl(const std::string& task, const std::shared_ptr<nn::Module>& m) {
        register_module(task, m);
    }
};
TORCH_MODULE(TaskWrap);

struct ResEncUNetConfig {
    int in_channels = 1;
    int num_classes = 2;            // surface: 2 (softmax); ink: 1 (sigmoid)
    int kernel = 3;
    std::string task = "surface";   // checkpoint task name (task_decoders.<task> / task_heads.<task>)
    bool task_head = false;         // false: surface (seg_layers in task_decoders.<task>)
                                    // true:  ink (shared_decoder + task_heads.<task> 1x1 conv)
    bool squeeze_excitation = true; // surface: scSE blocks; ink: plain residual blocks
    std::vector<int> features = {32, 64, 128, 256, 320, 320, 320};
    std::vector<int> blocks = {1, 3, 4, 6, 6, 6, 6};
    std::vector<int> strides = {1, 2, 2, 2, 2, 2, 2};
};

// The whole network. Surface: shared_encoder.* + task_decoders.<task>.* (seg_layers head).
// Ink: shared_encoder.* + shared_decoder.* + task_heads.<task>.* (single 1x1 conv). == checkpoint.
struct ResEncUNetImpl : nn::Module {
    Encoder shared_encoder{nullptr};
    Decoder decoder{nullptr};
    nn::Conv3d head{nullptr};
    bool task_head{false};
    explicit ResEncUNetImpl(const ResEncUNetConfig& c = {}) : task_head(c.task_head) {
        shared_encoder = register_module("shared_encoder",
            Encoder(c.in_channels, c.features, c.blocks, c.strides, c.kernel, c.squeeze_excitation));
        decoder = Decoder(c.features, c.num_classes, c.kernel, /*with_seg=*/!c.task_head);
        if (c.task_head) {
            register_module("shared_decoder", decoder);
            head = nn::Conv3d(nn::Conv3dOptions(c.features[0], c.num_classes, 1).bias(true));
            register_module("task_heads", TaskWrap(c.task, head.ptr()));
        } else {
            register_module("task_decoders", TaskWrap(c.task, decoder.ptr()));
        }
    }
    // Input (N,1,D,H,W) -> logits (N,num_classes,D,H,W).
    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor y = decoder(shared_encoder(x));
        return task_head ? head->forward(y) : y;
    }
};
TORCH_MODULE(ResEncUNet);

}  // namespace fenix::ml::nets
