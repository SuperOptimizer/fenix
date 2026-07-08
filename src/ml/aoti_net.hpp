// ml/aoti_net.hpp — AOTInductor `.pt2` package adapter for the predict machinery. Include ONLY
// from inference.cpp (the ML firewall TU — same rule as every torch header; see ml/CLAUDE.md).
// A `.pt2` package is STATIC-shape and torch-version/GPU-arch locked (matches the project's
// no-compat philosophy: rebuild per box with tools/ml-export/export_aoti.py, reject mismatches —
// the loader fails loudly). Captures torch.compile's fusion wins (norm/act/scSE fused around
// MIOpen/cuDNN convs) in the C++ path; no new dependency — the loader ships inside libtorch,
// so this works identically on CUDA and ROCm builds.
#pragma once

#include "core/core.hpp"

#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/torch.h>

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace fenix::ml {

// Walks like a torch net for run_predict_core / predict_surface_filled: net->forward([nb,1,P,P,P])
// -> logits [nb,C,P,P,P]. The package batch B and patch P are static (baked at export) — smaller
// nb is zero-padded to B and the output sliced back (the sliding-window tail batch), larger nb or
// a mismatched patch is a hard error. Input is cast to the export dtype (fp16) internally, so it
// composes with infer.hpp's opt.half path the same way TrtNet does.
class AotiNet {
  public:
    AotiNet* operator->() { return this; }

    static Expected<AotiNet> load(const std::string& path, torch::Device dev) {
        AotiNet n;
        try {
            n.loader_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
                path,
                "model",
                /*run_single_threaded=*/false,
                /*num_runners=*/1,
                dev.is_cuda() ? dev.index() : static_cast<torch::DeviceIndex>(-1));
        } catch (const std::exception& e) {
            return err(Errc::decode_error,
                       "aoti: package load failed (torch-version/GPU-arch locked — re-export with "
                       "tools/ml-export/export_aoti.py on THIS box): " +
                           path + ": " + e.what());
        }
        const auto md = n.loader_->get_metadata();
        auto geti = [&](const char* k) -> s64 {
            const auto it = md.find(k);
            return it == md.end() ? 0 : static_cast<s64>(std::strtoll(it->second.c_str(), nullptr, 10));
        };
        n.batch_ = geti("fenix_batch");
        n.patch_ = geti("fenix_patch");
        n.classes_ = geti("fenix_classes");
        if (n.batch_ <= 0 || n.patch_ <= 0 || n.classes_ <= 0)
            return err(Errc::decode_error,
                       "aoti: package missing fenix_batch/fenix_patch/fenix_classes metadata "
                       "(exported by tools/ml-export/export_aoti.py): " +
                           path);
        return n;
    }

    [[nodiscard]] s64 batch() const { return batch_; }
    [[nodiscard]] s64 patch() const { return patch_; }
    [[nodiscard]] s64 classes() const { return classes_; }

    // Throws on misuse (caller is inference.cpp, compiled with exceptions for the torch ABI).
    torch::Tensor forward(const torch::Tensor& x) {
        const s64 nb = x.size(0);
        TORCH_CHECK(nb <= batch_, "aoti: batch ", nb, " > package static batch ", batch_);
        TORCH_CHECK(x.size(2) == patch_, "aoti: patch ", x.size(2), " != package patch ", patch_);
        auto xi = x.to(torch::kHalf).contiguous();
        if (nb < batch_) {
            auto full = torch::zeros({batch_, 1, patch_, patch_, patch_}, xi.options());
            full.narrow(0, 0, nb).copy_(xi);
            xi = full;
        }
        // run() enqueues on torch's current stream: later torch ops (softmax/D2H) stay stream-ordered.
        auto y = loader_->run({xi}).at(0);
        return nb < batch_ ? y.narrow(0, 0, nb) : y;
    }
    // no-ops: a package is device/dtype-fixed at export time
    void to(torch::Device) {}
    void to(torch::ScalarType) {}
    void eval() {}

  private:
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> loader_;
    s64 batch_ = 0, patch_ = 0, classes_ = 0;
};

}  // namespace fenix::ml
