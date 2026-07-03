// ml/trt_engine.hpp — TensorRT engine adapter for the predict machinery. Include ONLY from
// inference.cpp (the ML firewall TU — same rule as every torch header; see ml/CLAUDE.md).
// A `.plan` engine is STATIC-shape and TRT-version/GPU-arch locked (matches the project's
// no-compat philosophy: rebuild per box, reject mismatches — deserialize fails loudly).
// Built by tools/ml-export/build_engine.py; measured 1.53x over eager fp16 at 256^3.
#pragma once

#include "core/core.hpp"

#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>
#include <cstdio>
#include <memory>
#include <NvInfer.h>
#include <string>
#include <vector>

namespace fenix::ml {

namespace detail {
class TrtLogger final : public nvinfer1::ILogger {
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kERROR)
            fenix::log(LogLevel::error, "tensorrt: {}", msg);
        else if (s == Severity::kWARNING)
            fenix::log(LogLevel::debug, "tensorrt: {}", msg);
    }
};
}  // namespace detail

// Walks like a torch net for run_predict_core: net->forward([nb,1,P,P,P]) -> logits
// [nb,C,P,P,P]. The engine batch B is static — smaller nb is zero-padded to B and the
// output sliced back (the sliding-window tail batch), larger nb is a hard error.
class TrtNet {
  public:
    TrtNet* operator->() { return this; }

    static Expected<TrtNet> load(const std::string& path) {
        TrtNet n;
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return err(Errc::not_found, "trt: cannot open " + path);
        std::fseek(f, 0, SEEK_END);
        std::vector<char> blob(static_cast<usize>(std::ftell(f)));
        std::fseek(f, 0, SEEK_SET);
        const bool rd = std::fread(blob.data(), 1, blob.size(), f) == blob.size();
        std::fclose(f);
        if (!rd) return err(Errc::io_error, "trt: short read " + path);

        n.runtime_.reset(nvinfer1::createInferRuntime(*n.logger_));
        if (!n.runtime_) return err(Errc::internal, "trt: createInferRuntime failed");
        n.engine_.reset(n.runtime_->deserializeCudaEngine(blob.data(), blob.size()));
        if (!n.engine_)
            return err(Errc::decode_error,
                       "trt: engine deserialize failed (TRT-version/GPU-arch locked — rebuild "
                       "with tools/ml-export/build_engine.py on THIS box): " +
                           path);
        n.ctx_.reset(n.engine_->createExecutionContext());
        if (!n.ctx_) return err(Errc::internal, "trt: createExecutionContext failed");

        for (s32 i = 0; i < n.engine_->getNbIOTensors(); ++i) {
            const char* nm = n.engine_->getIOTensorName(i);
            if (n.engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT)
                n.in_ = nm;
            else
                n.out_ = nm;
        }
        if (n.in_.empty() || n.out_.empty()) return err(Errc::decode_error, "trt: engine io tensors not found");
        const auto ishp = n.engine_->getTensorShape(n.in_.c_str());
        const auto oshp = n.engine_->getTensorShape(n.out_.c_str());
        if (ishp.nbDims != 5 || oshp.nbDims != 5)
            return err(Errc::unsupported, "trt: expected 5-D [B,1,P,P,P] engine io");
        n.batch_ = ishp.d[0];
        n.patch_ = ishp.d[2];
        n.classes_ = oshp.d[1];
        n.half_ = n.engine_->getTensorDataType(n.in_.c_str()) == nvinfer1::DataType::kHALF;
        return n;
    }

    [[nodiscard]] s64 batch() const { return batch_; }
    [[nodiscard]] s64 patch() const { return patch_; }
    [[nodiscard]] s64 classes() const { return classes_; }

    // Throws on misuse (caller is inference.cpp, compiled with exceptions for the torch ABI).
    torch::Tensor forward(const torch::Tensor& x) {
        const s64 nb = x.size(0);
        TORCH_CHECK(nb <= batch_, "trt: batch ", nb, " > engine static batch ", batch_);
        TORCH_CHECK(x.size(2) == patch_, "trt: patch ", x.size(2), " != engine patch ", patch_);
        const auto dt = half_ ? torch::kHalf : torch::kFloat32;
        auto xi = x.to(dt).contiguous();
        if (nb < batch_) {
            auto full = torch::zeros({batch_, 1, patch_, patch_, patch_}, xi.options());
            full.narrow(0, 0, nb).copy_(xi);
            xi = full;
        }
        auto y = torch::empty({batch_, classes_, patch_, patch_, patch_}, xi.options());
        ctx_->setTensorAddress(in_.c_str(), xi.data_ptr());
        ctx_->setTensorAddress(out_.c_str(), y.data_ptr());
        // enqueue on torch's current stream: later torch ops (softmax/D2H) are stream-ordered
        TORCH_CHECK(ctx_->enqueueV3(at::cuda::getCurrentCUDAStream().stream()), "trt: enqueueV3 failed");
        return nb < batch_ ? y.narrow(0, 0, nb) : y;
    }
    // no-ops: an engine is device/dtype-fixed at build time
    void to(torch::Device) {}
    void to(torch::ScalarType) {}
    void eval() {}

  private:
    struct Del {
        template <class T> void operator()(T* p) const { delete p; }
    };
    // heap-held: the runtime keeps a pointer to the logger, and TrtNet is moved out of load()
    std::unique_ptr<detail::TrtLogger> logger_ = std::make_unique<detail::TrtLogger>();
    std::unique_ptr<nvinfer1::IRuntime, Del> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, Del> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, Del> ctx_;
    std::string in_, out_;
    s64 batch_ = 0, patch_ = 0, classes_ = 0;
    bool half_ = true;
};

}  // namespace fenix::ml
