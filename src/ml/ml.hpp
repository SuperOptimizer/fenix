// ml/ml.hpp — neural inference via libtorch (C++). Firewalled behind FENIX_ML: the core
// build excludes torch entirely and this is a stub; an ML build (-DFENIX_ML=ON) pulls in
// torch_env + the network/weight headers. See ml/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#ifdef FENIX_ML
#include "ml/torch_env.hpp"
#include "ml/weights.hpp"
#include "ml/infer.hpp"
#include "ml/nets/resenc_unet.hpp"
#include "ml/nets/dinovol.hpp"
#include "ml/nets/resnet3d.hpp"
#include "io/nrrd.hpp"
#include "codec/archive.hpp"
#endif

namespace fenix::ml {

#ifdef FENIX_ML
// `fenix ml load-surface <surface.fxweights>` — build the C++ ResEnc-UNet, load the converted
// weights by name, report match completeness, and run one forward pass. This is the correctness
// gate for the from-scratch reimplementation (every architecture param must bind to a weight).
inline Expected<int> run_load_surface(std::string_view path) {
    nets::ResEncUNet net(nets::ResEncUNetConfig{});
    const auto dev = best_device();
    net->to(dev);
    net->eval();

    auto w = load_fxweights(std::string(path), dev);
    if (!w) return std::unexpected(w.error());

    std::vector<std::string> missing;
    const int matched = load_into(*net, *w, &missing);
    std::size_t n_params = 0;
    for (auto& kv : net->named_parameters()) { (void)kv; ++n_params; }

    fenix::log(LogLevel::info, "surface: {} file tensors, model has {} params, matched {}",
               w->size(), n_params, matched);
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "surface: {} model params had NO weight (arch mismatch):",
                   missing.size());
        for (std::size_t i = 0; i < missing.size() && i < 8; ++i)
            fenix::log(LogLevel::error, "  missing: {}", missing[i]);
        return fenix::err(Errc::decode_error, "surface weights/arch mismatch");
    }

    // Smoke forward: 64^3 (divisible by 2^6) to confirm the graph executes end-to-end.
    torch::NoGradGuard ng;
    auto x = torch::randn({1, 1, 64, 64, 64}, torch::TensorOptions().device(dev));
    auto y = net->forward(x);
    fenix::log(LogLevel::info, "surface: forward ok, out shape [{}]",
               [&] { std::string s; for (auto d : y.sizes()) s += std::to_string(d) + ","; return s; }());
    return 0;
}
#endif

#ifdef FENIX_ML
// `fenix ml run-raw <weights> <in.raw> <out.raw> <D> <H> <W>` — read a raw f32 volume, run the
// surface net (no normalization; raw logits out), write raw f32 [2,D,H,W]. Validation hook:
// lets a Python reference and this C++ reimpl be compared on identical input (Phase E).
inline Expected<int> run_raw(std::span<const std::string_view> a) {
    if (a.size() < 7) return fenix::err(Errc::invalid_argument,
        "usage: ml run-raw <weights> <in.raw> <out.raw> <D> <H> <W> [ink]");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    nets::ResEncUNetConfig cfg;
    if (a.size() >= 8 && a[7] == "ink") {
        cfg.task = "ink"; cfg.num_classes = 1; cfg.task_head = true; cfg.squeeze_excitation = false;
    }
    nets::ResEncUNet net(cfg);
    const auto dev = best_device();
    net->to(dev); net->eval();
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    std::vector<std::string> missing;
    load_into(*net, *w, &missing);
    if (!missing.empty()) return fenix::err(Errc::decode_error, "run-raw: weights/arch mismatch");

    std::FILE* fi = std::fopen(inpath.c_str(), "rb");
    if (!fi) return fenix::err(Errc::io_error, "run-raw: cannot open " + inpath);
    std::vector<float> buf(static_cast<std::size_t>(D * H * W));
    if (std::fread(buf.data(), sizeof(float), buf.size(), fi) != buf.size()) { std::fclose(fi); return fenix::err(Errc::io_error, "run-raw: short read"); }
    std::fclose(fi);

    torch::NoGradGuard ng;
    auto x = torch::from_blob(buf.data(), {1, 1, D, H, W}, torch::kFloat32).clone().to(dev);
    auto y = net->forward(x).to(torch::kCPU).contiguous();
    std::FILE* fo = std::fopen(outpath.c_str(), "wb");
    if (!fo) return fenix::err(Errc::io_error, "run-raw: cannot write " + outpath);
    std::fwrite(y.data_ptr<float>(), sizeof(float), static_cast<std::size_t>(y.numel()), fo);
    std::fclose(fo);
    fenix::log(LogLevel::info, "run-raw: wrote {} ({} floats)", outpath, y.numel());
    return 0;
}

// `fenix ml dino-raw <weights> <in.raw> <out.raw> <D> <H> <W>` — run the dinovol backbone, write
// raw f32 patch tokens (P, 864). Validation hook against reference_dino.py.
inline Expected<int> run_dino_raw(std::span<const std::string_view> a) {
    if (a.size() < 7) return fenix::err(Errc::invalid_argument,
        "usage: ml dino-raw <weights> <in.raw> <out.raw> <D> <H> <W>");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    nets::DinoVol net(nets::DinoVolConfig{});
    const auto dev = best_device();
    net->to(dev); net->eval();
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    std::vector<std::string> missing;
    load_into(*net, *w, &missing);
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "dino-raw: {} params unmatched (e.g. {})", missing.size(), missing[0]);
        return fenix::err(Errc::decode_error, "dino-raw: weights/arch mismatch");
    }
    std::FILE* fi = std::fopen(inpath.c_str(), "rb");
    if (!fi) return fenix::err(Errc::io_error, "dino-raw: cannot open " + inpath);
    std::vector<float> buf(static_cast<std::size_t>(D * H * W));
    if (std::fread(buf.data(), sizeof(float), buf.size(), fi) != buf.size()) { std::fclose(fi); return fenix::err(Errc::io_error, "short read"); }
    std::fclose(fi);
    torch::NoGradGuard ng;
    auto x = torch::from_blob(buf.data(), {1, 1, D, H, W}, torch::kFloat32).clone().to(dev);
    auto y = net->forward(x).to(torch::kCPU).contiguous();
    std::FILE* fo = std::fopen(outpath.c_str(), "wb");
    std::fwrite(y.data_ptr<float>(), sizeof(float), static_cast<std::size_t>(y.numel()), fo);
    std::fclose(fo);
    fenix::log(LogLevel::info, "dino-raw: wrote {} (shape [{},{}])", outpath, y.size(1), y.size(2));
    return 0;
}

// `fenix ml ink2d-raw <weights> <in.raw> <out.raw> <D> <H> <W>` — run the ResNet-152-3D ink
// model on a raw f32 segment subvolume, write the 2D ink logit. Validation hook vs reference.
inline Expected<int> run_ink2d_raw(std::span<const std::string_view> a) {
    if (a.size() < 7) return fenix::err(Errc::invalid_argument,
        "usage: ml ink2d-raw <weights> <in.raw> <out.raw> <D> <H> <W>");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    nets::ResNet3DInk net(true);
    const auto dev = best_device();
    net->to(dev); net->eval();
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    std::vector<std::string> missing;
    load_into(*net, *w, &missing);
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "ink2d-raw: {} params unmatched (e.g. {})", missing.size(), missing[0]);
        return fenix::err(Errc::decode_error, "ink2d-raw: weights/arch mismatch");
    }
    std::FILE* fi = std::fopen(inpath.c_str(), "rb");
    if (!fi) return fenix::err(Errc::io_error, "ink2d-raw: cannot open " + inpath);
    std::vector<float> buf(static_cast<std::size_t>(D * H * W));
    if (std::fread(buf.data(), sizeof(float), buf.size(), fi) != buf.size()) { std::fclose(fi); return fenix::err(Errc::io_error, "short read"); }
    std::fclose(fi);
    torch::NoGradGuard ng;
    auto x = torch::from_blob(buf.data(), {1, 1, D, H, W}, torch::kFloat32).clone().to(dev);
    auto y = net->forward(x).to(torch::kCPU).contiguous();
    std::FILE* fo = std::fopen(outpath.c_str(), "wb");
    std::fwrite(y.data_ptr<float>(), sizeof(float), static_cast<std::size_t>(y.numel()), fo);
    std::fclose(fo);
    fenix::log(LogLevel::info, "ink2d-raw: wrote {} ({} floats, out HxW)", outpath, y.numel());
    return 0;
}
#endif

#ifdef FENIX_ML
// Load a volume from .fxvol (codec archive) or .nrrd by extension.
inline Expected<Volume<f32>> load_volume(const std::string& path) {
    if (path.size() > 6 && path.substr(path.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::open(path);
        if (!a) return std::unexpected(a.error());
        return a->read_volume();
    }
    return io::read_nrrd(path);
}
#endif

#ifdef FENIX_ML
// Shared sliding-window predict: load volume, build/load the net, run inference, write output.
inline Expected<int> run_predict(std::span<const std::string_view> args, const char* name,
                                 nets::ResEncUNetConfig cfg, InferOptions opt) {
    if (args.size() < 3)
        return fenix::err(Errc::invalid_argument,
                          std::string("usage: ") + name + " <in.fxvol|.nrrd> <weights.fxweights> "
                          "<out.fxvol|.nrrd> [patch] [overlap] [tta]");
    const std::string inpath(args[0]), wpath(args[1]), outpath(args[2]);
    if (args.size() >= 4) opt.patch = std::stoi(std::string(args[3]));
    if (args.size() >= 5) opt.overlap = std::stod(std::string(args[4]));
    if (args.size() >= 6) opt.tta = std::stoi(std::string(args[5]));

    auto vol = load_volume(inpath);
    if (!vol) return std::unexpected(vol.error());
    const auto d = vol->dims();
    const int tta_n = opt.tta <= 1 ? 1 : (opt.tta < 48 ? opt.tta : 48);
    fenix::log(LogLevel::info, "{}: input {}x{}x{} (ZYX), patch={} overlap={} tta={}", name, d.z, d.y, d.x,
               opt.patch, opt.overlap, tta_n);

    nets::ResEncUNet net(cfg);
    const auto dev = best_device();
    net->to(dev); net->eval();
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    std::vector<std::string> missing;
    load_into(*net, *w, &missing);
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "{}: {} model params unmatched (e.g. {})", name, missing.size(), missing[0]);
        return fenix::err(Errc::decode_error, std::string(name) + ": weights/arch mismatch");
    }
    fenix::log(LogLevel::info, "{}: model loaded on {}", name, dev.str());

    auto prob = predict_surface(vol->view(), net, dev, opt);
    if (!prob) return std::unexpected(prob.error());

    if (outpath.size() > 6 && outpath.substr(outpath.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::create(outpath, d, codec::DctParams{});
        if (!a) return std::unexpected(a.error());
        if (auto r = a->write_volume(prob->view()); !r) return std::unexpected(r.error());
        if (auto r = a->close(); !r) return std::unexpected(r.error());
    } else {
        if (auto r = io::write_nrrd(outpath, prob->view()); !r) return std::unexpected(r.error());
    }
    fenix::log(LogLevel::info, "{}: wrote {}", name, outpath);
    return 0;
}
#endif

// `fenix predict-surface <in> <surface.fxweights> <out> [patch] [overlap]`
inline Expected<int> run_predict_surface([[maybe_unused]] std::span<const std::string_view> args) {
#ifdef FENIX_ML
    nets::ResEncUNetConfig cfg;  // defaults = surface (task_decoders.surface, 2-class softmax)
    InferOptions opt;            // zscore, softmax channel 1
    return run_predict(args, "predict-surface", cfg, opt);
#else
    return stage_unimplemented("predict-surface");
#endif
}

// `fenix predict-ink <in> <ink.fxweights> <out> [patch] [overlap]`
inline Expected<int> run_predict_ink([[maybe_unused]] std::span<const std::string_view> args) {
#ifdef FENIX_ML
    nets::ResEncUNetConfig cfg;
    cfg.task = "ink";
    cfg.num_classes = 1;
    cfg.task_head = true;        // shared_decoder + task_heads.ink (single 1x1 conv)
    cfg.squeeze_excitation = false;  // ink encoder has plain residual blocks (no scSE)
    InferOptions opt;
    opt.sigmoid = true;          // 1-channel logit -> sigmoid
    opt.norm = Norm::pct_minmax; // percentile (0.5/99.5) min-max
    return run_predict(args, "predict-ink", cfg, opt);
#else
    return stage_unimplemented("predict-ink");
#endif
}

// `fenix ml [info | load-surface <weights> | run-raw ...]`.
inline Expected<int> run([[maybe_unused]] std::span<const std::string_view> args,
                         [[maybe_unused]] Context& ctx) {
#ifdef FENIX_ML
    if (!args.empty() && args[0] == "load-surface") {
        if (args.size() < 2) return fenix::err(Errc::invalid_argument, "usage: ml load-surface <surface.fxweights>");
        return run_load_surface(args[1]);
    }
    if (!args.empty() && args[0] == "run-raw") return run_raw(args);
    if (!args.empty() && args[0] == "dino-raw") return run_dino_raw(args);
    if (!args.empty() && args[0] == "ink2d-raw") return run_ink2d_raw(args);
    fenix::log(LogLevel::info, "ml: {}", device_summary());
    const auto dev = best_device();
    const auto t = torch::randn({256, 256}, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
    const float s = torch::matmul(t, t).sum().item<float>();
    fenix::log(LogLevel::info, "ml: self-test matmul on {} ok (sum={:.3f})", dev.str(), s);
    return 0;
#else
    return stage_unimplemented("ml");
#endif
}

}  // namespace fenix::ml

FENIX_REGISTER_STAGE(ml, "ml stage (libtorch inference)", ::fenix::ml::run)

namespace {
// Hyphenated subcommand names (the macro only makes identifier-named stages).
[[maybe_unused]] const int fenix_stage_predict_surface = ::fenix::register_stage(
    ::fenix::Stage{"predict-surface", "surface-prediction inference (3D ResEnc-UNet)",
                   [](std::span<const std::string_view> a, ::fenix::Context&) {
                       return ::fenix::ml::run_predict_surface(a);
                   }});
[[maybe_unused]] const int fenix_stage_predict_ink = ::fenix::register_stage(
    ::fenix::Stage{"predict-ink", "ink-detection inference (3D ResEnc-UNet, DINO-guided)",
                   [](std::span<const std::string_view> a, ::fenix::Context&) {
                       return ::fenix::ml::run_predict_ink(a);
                   }});
}  // namespace
