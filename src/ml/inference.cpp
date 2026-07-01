// ml/inference.cpp — THE libtorch translation unit: the ONLY place <torch/torch.h> is parsed. Compiled
// only for FENIX_ML builds; it defines the torch-free entry points declared in ml/ml_api.hpp. Isolating
// libtorch here (instead of inline in ml.hpp, which the driver pulled in) keeps the driver and every other
// TU torch-free — the unity driver TU drops from ~7 min to seconds. See ADR 0008 (build firewall).
#include "ml/ml_api.hpp"

#ifdef FENIX_ML

#include "codec/archive.hpp"
#include "io/nrrd.hpp"
#include "ml/infer.hpp"
#include "ml/nets/dinovol.hpp"
#include "ml/nets/resenc_unet.hpp"
#include "ml/nets/resnet3d.hpp"
#include "ml/torch_env.hpp"
#include "ml/weights.hpp"

#include <optional>

namespace fenix::ml {

// `fenix ml load-surface <surface.fxweights>` — build the C++ ResEnc-UNet, load the converted
// weights by name, report match completeness, and run one forward pass. This is the correctness
// gate for the from-scratch reimplementation (every architecture param must bind to a weight).
static Expected<int> run_load_surface(std::string_view path) {
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

// `fenix ml run-raw <weights> <in.raw> <out.raw> <D> <H> <W>` — read a raw f32 volume, run the
// surface net (no normalization; raw logits out), write raw f32 [2,D,H,W]. Validation hook.
static Expected<int> run_raw(std::span<const std::string_view> a) {
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
static Expected<int> run_dino_raw(std::span<const std::string_view> a) {
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
static Expected<int> run_ink2d_raw(std::span<const std::string_view> a) {
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

// Load a volume from .fxvol (codec archive) or .nrrd by extension.
static Expected<Volume<f32>> load_volume(const std::string& path) {
    if (path.size() > 6 && path.substr(path.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::open(path);
        if (!a) return std::unexpected(a.error());
        return a->read_volume();
    }
    return io::read_nrrd(path);
}

// Shared sliding-window predict: load volume, build/load the net, run inference, write output.
static Expected<int> run_predict(std::span<const std::string_view> args, const char* name,
                                 nets::ResEncUNetConfig cfg, InferOptions opt) {
    if (args.size() < 3)
        return fenix::err(Errc::invalid_argument,
                          std::string("usage: ") + name + " <in.fxvol|.nrrd> <weights.fxweights> "
                          "<out.fxvol|.nrrd> [patch] [overlap] [tta]");
    const std::string inpath(args[0]), wpath(args[1]), outpath(args[2]);
    if (args.size() >= 4) opt.patch = std::stoi(std::string(args[3]));
    if (args.size() >= 5) opt.overlap = std::stod(std::string(args[4]));
    if (args.size() >= 6) opt.tta = std::stoi(std::string(args[5]));

    init_torch_threads();  // clamp torch CPU pools to the cgroup budget before any op (container safety)

    // Input read strategy — decode the .fxvol ONCE into a dense NATIVE-u8 volume (8 GiB for a 2048³, NOT the
    // 34 GiB f32 slab), then gather patches from that flat array (a plain copy, no per-patch DCT decode, no
    // cache locks). This is far faster than streaming/re-decoding overlapping tiles per patch, and still
    // never widens u8→f32 for storage. (Streaming/out-of-core stays available via the archive's block cache
    // for volumes that don't fit in RAM — not needed here.)
    const bool fxvol_in = inpath.size() > 6 && inpath.substr(inpath.size() - 6) == ".fxvol";
    Volume<u8> vol_u8;
    Volume<f32> nrrd_vol;
    bool u8_src = false;
    Extent3 d{};
    if (fxvol_in) {
        auto a = codec::VolumeArchive::open(inpath);
        if (!a) return std::unexpected(a.error());
        if (a->src_dtype() == codec::DType::u8) {
            auto v = a->read_volume_as<u8>(0);  // parallel one-time decode → dense u8
            if (!v) return std::unexpected(v.error());
            vol_u8 = std::move(*v);
            d = vol_u8.dims();
            u8_src = true;
        } else {
            auto v = a->read_volume(0);
            if (!v) return std::unexpected(v.error());
            nrrd_vol = std::move(*v);
            d = nrrd_vol.dims();
        }
    } else {
        auto vol = load_volume(inpath);
        if (!vol) return std::unexpected(vol.error());
        nrrd_vol = std::move(*vol);
        d = nrrd_vol.dims();
    }
    const int tta_n = opt.tta <= 1 ? 1 : (opt.tta < 48 ? opt.tta : 48);
    fenix::log(LogLevel::info, "{}: input {}x{}x{} (ZYX), patch={} overlap={} tta={} src={}", name, d.z, d.y,
               d.x, opt.patch, opt.overlap, tta_n, u8_src ? "fxvol(dense u8)" : "dense f32");

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

    Expected<Volume<f32>> prob = fenix::err(Errc::unsupported, "unset");
    if (u8_src) {
        // Gather from the dense u8 volume (widen to f32 per patch, parallel over z) — a plain array copy, no
        // decode/locks in the hot loop.
        VolumeView<const u8> uv = vol_u8.view();
        prob = predict_surface_filled(
            d,
            [uv](s64 z0, s64 y0, s64 x0, int P, float* out) {
                parallel_for(0, P, [&](s64 z) {
                    for (int y = 0; y < P; ++y)
                        for (int x = 0; x < P; ++x)
                            out[(static_cast<std::size_t>(z) * P + y) * P + x] =
                                static_cast<f32>(uv.at_clamped(z0 + z, y0 + y, x0 + x));
                });
            },
            net, dev, opt);
    } else {
        prob = predict_surface(nrrd_vol.view(), net, dev, opt);
    }
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

// ---- torch-free entry points (declared in ml/ml_api.hpp) -------------------------------------------

// `fenix predict-surface <in> <surface.fxweights> <out> [patch] [overlap]`
Expected<int> run_predict_surface(std::span<const std::string_view> args) {
    nets::ResEncUNetConfig cfg;  // defaults = surface (task_decoders.surface, 2-class softmax)
    InferOptions opt;            // zscore, softmax channel 1
    return run_predict(args, "predict-surface", cfg, opt);
}

// `fenix predict-ink <in> <ink.fxweights> <out> [patch] [overlap]`
Expected<int> run_predict_ink(std::span<const std::string_view> args) {
    nets::ResEncUNetConfig cfg;
    cfg.task = "ink";
    cfg.num_classes = 1;
    cfg.task_head = true;        // shared_decoder + task_heads.ink (single 1x1 conv)
    cfg.squeeze_excitation = false;  // ink encoder has plain residual blocks (no scSE)
    InferOptions opt;
    opt.sigmoid = true;          // 1-channel logit -> sigmoid
    opt.norm = Norm::pct_minmax; // percentile (0.5/99.5) min-max
    return run_predict(args, "predict-ink", cfg, opt);
}

// `fenix ml [info | load-surface <weights> | run-raw ...]`.
Expected<int> run(std::span<const std::string_view> args, Context&) {
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
}

}  // namespace fenix::ml

#endif  // FENIX_ML
