// ml/inference.cpp — THE libtorch translation unit: the ONLY place <torch/torch.h> is parsed. Compiled
// only for FENIX_ML builds; it defines the torch-free entry points declared in ml/ml_api.hpp. Isolating
// libtorch here (instead of inline in ml.hpp, which the driver pulled in) keeps the driver and every other
// TU torch-free — the unity driver TU drops from ~7 min to seconds. See ADR 0008 (build firewall).
#include "ml/ml_api.hpp"

#ifdef FENIX_ML

#include "codec/archive.hpp"
#include "io/surface.hpp"
#include "io/zarr.hpp"
#include "ml/infer.hpp"
#include "ml/nets/dinovol.hpp"
#include "ml/nets/resenc_unet.hpp"
#include "ml/surface_index.hpp"
#ifdef FENIX_TRT
#include "ml/trt_engine.hpp"
#endif
#include "io/jpeg.hpp"
#include "ml/nets/resnet3d.hpp"
#include "ml/torch_env.hpp"
#include "ml/weights.hpp"

#include <torch/script.h>  // torch::jit::load — the .ts student path

#include <charconv>
#include <filesystem>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <system_error>

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
    for (auto& kv : net->named_parameters()) {
        (void)kv;
        ++n_params;
    }

    fenix::log(
        LogLevel::info, "surface: {} file tensors, model has {} params, matched {}", w->size(), n_params, matched);
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "surface: {} model params had NO weight (arch mismatch):", missing.size());
        for (std::size_t i = 0; i < missing.size() && i < 8; ++i)
            fenix::log(LogLevel::error, "  missing: {}", missing[i]);
        return fenix::err(Errc::decode_error, "surface weights/arch mismatch");
    }

    // Smoke forward: 64^3 (divisible by 2^6) to confirm the graph executes end-to-end.
    torch::NoGradGuard ng;
    auto x = torch::randn({1, 1, 64, 64, 64}, torch::TensorOptions().device(dev));
    auto y = net->forward(x);
    fenix::log(LogLevel::info, "surface: forward ok, out shape [{}]", [&] {
        std::string s;
        for (auto d : y.sizes()) s += std::to_string(d) + ",";
        return s;
    }());
    return 0;
}

// `fenix ml run-raw <weights> <in.raw> <out.raw> <D> <H> <W>` — read a raw f32 volume, run the
// surface net (no normalization; raw logits out), write raw f32 [2,D,H,W]. Validation hook.
static Expected<int> run_raw(std::span<const std::string_view> a) {
    if (a.size() < 7)
        return fenix::err(Errc::invalid_argument, "usage: ml run-raw <weights> <in.raw> <out.raw> <D> <H> <W> [ink]");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    nets::ResEncUNetConfig cfg;
    if (a.size() >= 8 && a[7] == "ink") {
        cfg.task = "ink";
        cfg.num_classes = 1;
        cfg.task_head = true;
        cfg.squeeze_excitation = false;
    }
    nets::ResEncUNet net(cfg);
    const auto dev = best_device();
    net->to(dev);
    net->eval();
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    std::vector<std::string> missing;
    load_into(*net, *w, &missing);
    if (!missing.empty()) return fenix::err(Errc::decode_error, "run-raw: weights/arch mismatch");

    std::FILE* fi = std::fopen(inpath.c_str(), "rb");
    if (!fi) return fenix::err(Errc::io_error, "run-raw: cannot open " + inpath);
    std::vector<float> buf(static_cast<std::size_t>(D * H * W));
    if (std::fread(buf.data(), sizeof(float), buf.size(), fi) != buf.size()) {
        std::fclose(fi);
        return fenix::err(Errc::io_error, "run-raw: short read");
    }
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
    if (a.size() < 7)
        return fenix::err(Errc::invalid_argument, "usage: ml dino-raw <weights> <in.raw> <out.raw> <D> <H> <W>");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    nets::DinoVol net(nets::DinoVolConfig{});
    const auto dev = best_device();
    net->to(dev);
    net->eval();
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
    if (std::fread(buf.data(), sizeof(float), buf.size(), fi) != buf.size()) {
        std::fclose(fi);
        return fenix::err(Errc::io_error, "short read");
    }
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

// `fenix ml ink2d-raw <weights> <in.raw> <out.raw> <D> <H> <W> [r50]` — run the 2D-ink model
// (ResNet-152-3D-decoder, or resnet3d-50 with `r50`) on a raw f32 subvolume, write the 2D ink
// logit. Validation hook vs reference.
static Expected<int> run_ink2d_raw(std::span<const std::string_view> a) {
    if (a.size() < 7)
        return fenix::err(Errc::invalid_argument, "usage: ml ink2d-raw <weights> <in.raw> <out.raw> <D> <H> <W> [r50]");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    const bool r50 = a.size() >= 8 && a[7] == "r50";
    const auto dev = best_device();
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    nets::ResNet3DInk net152{nullptr};
    nets::ResNet3DInk50 net50{nullptr};
    std::function<torch::Tensor(torch::Tensor)> fwd;
    std::vector<std::string> missing;
    if (r50) {
        net50 = nets::ResNet3DInk50(w->count("normalization.weight") > 0);
        net50->to(dev);
        net50->eval();
        load_into(*net50, *w, &missing);
        fwd = [&net50](torch::Tensor t) { return net50->forward(t); };
    } else {
        net152 = nets::ResNet3DInk(true);
        net152->to(dev);
        net152->eval();
        load_into(*net152, *w, &missing);
        fwd = [&net152](torch::Tensor t) { return net152->forward(t); };
    }
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "ink2d-raw: {} params unmatched (e.g. {})", missing.size(), missing[0]);
        return fenix::err(Errc::decode_error, "ink2d-raw: weights/arch mismatch");
    }
    std::FILE* fi = std::fopen(inpath.c_str(), "rb");
    if (!fi) return fenix::err(Errc::io_error, "ink2d-raw: cannot open " + inpath);
    std::vector<float> buf(static_cast<std::size_t>(D * H * W));
    if (std::fread(buf.data(), sizeof(float), buf.size(), fi) != buf.size()) {
        std::fclose(fi);
        return fenix::err(Errc::io_error, "short read");
    }
    std::fclose(fi);
    torch::NoGradGuard ng;
    auto x = torch::from_blob(buf.data(), {1, 1, D, H, W}, torch::kFloat32).clone().to(dev);
    auto y = fwd(x).to(torch::kCPU).contiguous();
    std::FILE* fo = std::fopen(outpath.c_str(), "wb");
    std::fwrite(y.data_ptr<float>(), sizeof(float), static_cast<std::size_t>(y.numel()), fo);
    std::fclose(fo);
    fenix::log(LogLevel::info, "ink2d-raw: wrote {} ({} floats, out HxW)", outpath, y.numel());
    return 0;
}

// Load a volume from a .fxvol codec archive (the only volume container fenix reads/writes).
static Expected<Volume<f32>> load_volume(const std::string& path) {
    if (!(path.size() > 6 && path.substr(path.size() - 6) == ".fxvol"))
        return err(Errc::unsupported, "expected a .fxvol volume, got " + path);
    auto a = codec::VolumeArchive::open(path);
    if (!a) return std::unexpected(a.error());
    return a->read_volume();
}

// Shared sliding-window predict: load volume, build/load the net, run inference, write output.
// Adapter making a TorchScript module walk like a torch::nn ModuleHolder in the predict
// templates (they call net->forward(x)).
struct JitNet {
    torch::jit::Module m;
    JitNet* operator->() { return this; }
    torch::Tensor forward(const torch::Tensor& x) { return m.forward({x}).toTensor(); }
    void to(torch::Device d) { m.to(d); }
    void to(torch::ScalarType t) { m.to(t); }
    void eval() { m.eval(); }
};

// The net-generic back half of run_predict: checkpoint identity, TTA/batched dispatch, output
// write. Net is anything with net->forward(Tensor)->Tensor (nets::ResEncUNet or the TorchScript
// adapter below) — the student export path (Round E) loads .ts modules through the same machinery.
template <class Net>
static Expected<int> run_predict_core(const char* name,
                                      const std::string& inpath,
                                      const std::string& wpath,
                                      const std::string& outpath,
                                      InferOptions opt,
                                      Net& net,
                                      torch::Device dev,
                                      Extent3 d,
                                      Volume<u8>& vol_u8,
                                      Volume<f32>& dense_vol,
                                      bool u8_src) {
    const bool prof0 = std::getenv("FENIX_INFER_PROFILE") != nullptr;
    auto clk0 = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td = clk0();
    // Checkpoint identity (findings: a resume must never silently blend a different model/input/task).
    // weights_hash is a content hash (not path/mtime — weights get copied/re-downloaded between boxes,
    // which would spuriously invalidate a multi-hour checkpoint on mtime alone) computed by streaming the
    // .fxweights file once; input_hash is a cheap path-based identity (hashing a multi-GB volume's full
    // content on every run would itself be a material cost, so this catches the common cases — same
    // output path re-run with a different input file — without adding a decode pass).
    {
        std::FILE* wf = std::fopen(wpath.c_str(), "rb");
        if (wf) {
            u64 h = 0;
            std::vector<u8> chunk(1 << 20);
            for (;;) {
                const std::size_t n = std::fread(chunk.data(), 1, chunk.size(), wf);
                if (n == 0) break;
                h = hash64({chunk.data(), n}, h);
            }
            std::fclose(wf);
            opt.weights_hash = h;
        }
        opt.input_hash = hash64({reinterpret_cast<const u8*>(inpath.data()), inpath.size()});
    }

    Expected<Volume<f32>> prob = fenix::err(Errc::unsupported, "unset");
    Volume<f32> f32src;  // widened u8 source, kept alive for the multi-scale/rotation/noise/offset path
    const bool ens = !opt.scales.empty() || !opt.rots.empty() || opt.noise > 0 || opt.offsets > 0;
    if (ens) {
        // Ensemble TTA (scales/rots/noise/offsets) runs through the f32 wrapper — widen the dense u8 once
        // if that's the input.
        VolumeView<const f32> srcv;
        if (u8_src) {
            f32src = Volume<f32>(d);
            auto sv = f32src.view();
            VolumeView<const u8> uv = vol_u8.view();
            parallel_for(0, d.z, [&](s64 z) {
                for (s64 y = 0; y < d.y; ++y)
                    for (s64 x = 0; x < d.x; ++x) sv(z, y, x) = static_cast<f32>(uv(z, y, x));
            });
            srcv = f32src.view();
        } else {
            srcv = dense_vol.view();
        }
        fenix::log(LogLevel::info,
                   "{}: ensemble TTA — {} scales, {} rots, {} noise (s={:.3g}), {} offsets",
                   name,
                   opt.scales.empty() ? 1 : opt.scales.size(),
                   opt.rots.empty() ? 1 : opt.rots.size(),
                   opt.noise,
                   opt.noise_sigma,
                   opt.offsets);
        prob = predict_surface_tta(srcv, net, dev, opt);
    } else if (u8_src) {
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
            net,
            dev,
            opt);
    } else {
        prob = predict_surface(dense_vol.view(), net, dev, opt);
    }
    if (!prob) return std::unexpected(prob.error());

    // The dense source is done — release it before the output encode allocates its buffers
    // (1 GiB at 1024³, 8 GiB at 2048³ off the peak during the write phase).
    vol_u8 = Volume<u8>();
    dense_vol = Volume<f32>();
    f32src = Volume<f32>();

    td = clk0();
    // ML predictions write a u8-native .fxvol at q=32: the [0,1] probability is SCALED to [0,255] and
    // stored as u8, so the codec's q (an ABSOLUTE step calibrated for [0,255] CT data) means the same
    // thing it does everywhere else — q=32 on raw [0,1] probs would dead-zone the whole field to zero.
    // u8-native (never f32 on disk), never NRRD. q override via FENIX_PREDICT_Q; default 32.
    f32 pq = 32.0f;
    if (const char* e = std::getenv("FENIX_PREDICT_Q")) {
        const f32 v = std::atof(e);
        if (v > 0) pq = v;
    }
    {
        Volume<u8> pred8(d);
        auto pvv = prob->view();
        auto p8 = pred8.view();
        parallel_for_z(d, [&](s64 z) {
            for (s64 y = 0; y < d.y; ++y)
                for (s64 x = 0; x < d.x; ++x)
                    p8(z, y, x) = static_cast<u8>(std::clamp(pvv(z, y, x), 0.0f, 1.0f) * 255.0f + 0.5f);
        });
        auto a = codec::VolumeArchive::create(outpath, d, codec::DctParams{.q = pq});
        if (!a) return std::unexpected(a.error());
        if (auto r = a->template write_volume<u8>(p8); !r) return std::unexpected(r.error());
        if (auto r = a->close(); !r) return std::unexpected(r.error());
    }
    if (prof0) fenix::log(LogLevel::info, "T write-output: {:.1f}s", clk0() - td);
    fenix::log(LogLevel::info, "{}: wrote {}", name, outpath);
    return 0;
}

static Expected<int>
run_predict(std::span<const std::string_view> args, const char* name, nets::ResEncUNetConfig cfg, InferOptions opt) {
    if (args.size() < 3)
        return fenix::err(Errc::invalid_argument,
                          std::string("usage: ") + name +
                              " <in.fxvol> <weights.fxweights> "
                              "<out.fxvol> [patch] [overlap] [tta] [batch]");
    const std::string inpath(args[0]), wpath(args[1]), outpath(args[2]);
    // Positional numeric slots. A keyword token (contains '=', e.g. "scales=0.8,1.0") in any of these
    // slots must NOT reach std::stoi/stod — that throws std::invalid_argument, which is uncaught here
    // (this whole call chain runs in a FENIX_ML/exceptions build but the driver above it is
    // -fno-exceptions) and would std::terminate the process on a usage typo instead of reporting it.
    auto parse_int = [](std::string_view s, int& out) -> Expected<void> {
        int v = 0;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
            return fenix::err(Errc::invalid_argument, "predict: not an integer: " + std::string(s));
        out = v;
        return {};
    };
    auto parse_double = [](std::string_view s, double& out) -> Expected<void> {
        double v = 0;
        const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
            return fenix::err(Errc::invalid_argument, "predict: not a number: " + std::string(s));
        out = v;
        return {};
    };
    if (args.size() >= 4 && args[3].find('=') == std::string_view::npos) {
        if (auto r = parse_int(args[3], opt.patch); !r) return std::unexpected(r.error());
    }
    if (args.size() >= 5 && args[4].find('=') == std::string_view::npos) {
        if (auto r = parse_double(args[4], opt.overlap); !r) return std::unexpected(r.error());
    }
    if (args.size() >= 6 && args[5].find('=') == std::string_view::npos) {
        if (auto r = parse_int(args[5], opt.tta); !r) return std::unexpected(r.error());
    }
    // arg[6] positional [batch] (patches per GPU forward). Skipped if it's a keyword token (e.g. scales=).
    bool batch_explicit = false;
    if (args.size() >= 7 && args[6].find('=') == std::string_view::npos) {
        if (auto r = parse_int(args[6], opt.batch); !r) return std::unexpected(r.error());
        batch_explicit = true;
    }
    // GPU auto-profile: when batch isn't explicit, pick the MEASURED sweet spot for the card
    // (configs/gpu/*.toml documents the sweeps). Saturation, not VRAM, sets the ceiling:
    //   RTX 5090 32 GB      -> batch=3  (measured 2026-06-30)
    //   RTX PRO 6000 96 GB  -> batch=6  (measured 2026-07-02: b6 63.6s == b12 63.5s on 1024^3; b24 OOM)
    // Unknown cards keep the conservative default (3). Override: positional [batch].
    // Device name via nvidia-smi (one popen at startup): the libtorch C++ device-properties API
    // needs CUDA toolkit headers this build intentionally lacks (we LINK the torch wheel, never
    // compile CUDA — see ml CLAUDE.md). Fails soft: unknown/absent -> keep the default.
    if (!batch_explicit && torch::cuda::is_available()) {
        std::string dev_name;
        if (std::FILE* pf = ::popen("nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null", "r")) {
            char buf[256] = {};
            if (std::fgets(buf, sizeof buf, pf)) dev_name = buf;
            ::pclose(pf);
        }
        if (dev_name.find("RTX PRO 6000") != std::string::npos)
            opt.batch = 6;
        else if (dev_name.find("5090") != std::string::npos)
            opt.batch = 3;
        if (!dev_name.empty())
            fenix::log(
                LogLevel::debug, "{}: gpu '{}' -> batch={}", name, dev_name.substr(0, dev_name.find('\n')), opt.batch);
    }
    // scales=a,b,c / rots=d1,d2,... (scanned anywhere in args) — multi-scale TTA mean-fuse / arbitrary
    // z-rotation TTA in degrees. A single scale = base rescale.
    std::string band_paths;          // band=<seg.fxsurf[,seg2.fxsurf...]> — predict only near these surfaces
    double band_r = 640;             // band_r=<vox> tile dilation; must cover the training sampler's max
                                     // offset (spread 1.5*patch + jitter + patch/2) so every sampled patch
                                     // lands on predicted teacher voxels
    std::vector<double> band_off;    // band_off=z,y,x — the crop origin: fxsurf coords are GLOBAL volume
                                     // voxels, the input .fxvol is a crop
    std::optional<Error> parse_err;  // first error wins; keep parsing to report all consumed tokens
    auto note_err = [&](Expected<void> r) {
        if (!r && !parse_err) parse_err = r.error();
    };
    auto parse_list = [&](std::string_view list, std::vector<double>& out) {
        for (std::size_t p = 0; p < list.size();) {
            const std::size_t c = list.find(',', p);
            const std::string_view tok = list.substr(p, c == std::string_view::npos ? std::string_view::npos : c - p);
            if (!tok.empty()) {
                double v = 0;
                note_err(parse_double(tok, v));
                out.push_back(v);
            }
            if (c == std::string_view::npos) break;
            p = c + 1;
        }
    };
    for (auto a : args) {
        if (a.size() > 7 && a.substr(0, 7) == "scales=") parse_list(a.substr(7), opt.scales);
        if (a.size() > 5 && a.substr(0, 5) == "rots=") parse_list(a.substr(5), opt.rots);
        if (a.size() > 6 && a.substr(0, 6) == "noise=") note_err(parse_int(a.substr(6), opt.noise));
        if (a.size() > 7 && a.substr(0, 7) == "nsigma=") note_err(parse_double(a.substr(7), opt.noise_sigma));
        if (a.size() > 8 && a.substr(0, 8) == "offsets=") note_err(parse_int(a.substr(8), opt.offsets));
        if (a.size() > 5 && a.substr(0, 5) == "band=") band_paths = std::string(a.substr(5));
        if (a.size() > 7 && a.substr(0, 7) == "band_r=") note_err(parse_double(a.substr(7), band_r));
        if (a.size() > 9 && a.substr(0, 9) == "band_off=") parse_list(a.substr(9), band_off);
        if (a.size() > 5 && a.substr(0, 5) == "ckpt=") opt.ckpt_path = std::string(a.substr(5));
        if (a.size() > 11 && a.substr(0, 11) == "ckpt_every=") {
            long v = 0;
            const auto tail = a.substr(11);
            if (auto r = std::from_chars(tail.data(), tail.data() + tail.size(), v);
                r.ec != std::errc{} || r.ptr != tail.data() + tail.size())
                note_err(fenix::err(Errc::invalid_argument, "predict: not an integer: " + std::string(tail)));
            else
                opt.ckpt_every = v;
        }
    }
    if (parse_err) return std::unexpected(*parse_err);
    // Band filter: load the surfaces (crop-local coords via band_off), build the R-tree index, and
    // keep only tiles whose band_r-dilated box touches a surface. See InferOptions::tile_filter.
    if (!band_paths.empty()) {
        if (band_off.size() != 0 && band_off.size() != 3)
            return fenix::err(Errc::invalid_argument, "predict: band_off wants z,y,x");
        const Vec3f off =
            band_off.size() == 3
                ? Vec3f{static_cast<f32>(band_off[0]), static_cast<f32>(band_off[1]), static_cast<f32>(band_off[2])}
                : Vec3f{0, 0, 0};
        auto surfs = std::make_shared<std::vector<Surface>>();
        for (std::size_t p = 0; p < band_paths.size();) {
            const std::size_t c = band_paths.find(',', p);
            const std::string one = band_paths.substr(p, c == std::string::npos ? std::string::npos : c - p);
            if (!one.empty()) {
                auto s = io::read_fxsurf(one);
                if (!s) return std::unexpected(s.error());
                for (auto& cc : s->coord) cc = cc - off;
                surfs->push_back(std::move(*s));
            }
            if (c == std::string::npos) break;
            p = c + 1;
        }
        auto ptrs = std::make_shared<std::vector<const Surface*>>();
        for (auto& s : *surfs) ptrs->push_back(&s);
        auto idx = std::make_shared<VolumeSurfaceIndex>(std::span<const Surface* const>(*ptrs));
        const f32 r = static_cast<f32>(band_r);
        opt.tile_filter = [surfs, ptrs, idx, r](s64 z0, s64 y0, s64 x0, s64 P) {
            const geom::Box3f q{static_cast<f32>(z0) - r,
                                static_cast<f32>(z0 + P) + r,
                                static_cast<f32>(y0) - r,
                                static_cast<f32>(y0 + P) + r,
                                static_cast<f32>(x0) - r,
                                static_cast<f32>(x0 + P) + r};
            return !idx->query(q).empty();
        };
        fenix::log(LogLevel::info, "{}: band filter over {} surface(s), r={}", name, surfs->size(), band_r);
    }
    // ckpt_every=0 (or negative) would hit `n_tiles % opt.ckpt_every` -> SIGFPE; clamp to a sane minimum.
    opt.ckpt_every = std::max<long>(1, opt.ckpt_every);
    // Resumable by DEFAULT: checkpoint next to the output as <out>.ckpt (auto-resumes if the run is
    // re-launched with the same args). `ckpt=off` disables it; `ckpt=<path>` overrides the location.
    if (opt.ckpt_path.empty())
        opt.ckpt_path = outpath + ".ckpt";
    else if (opt.ckpt_path == "off")
        opt.ckpt_path.clear();

    init_torch_threads();  // clamp torch CPU pools to the cgroup budget before any op (container safety)

    // Input read strategy — decode the .fxvol ONCE into a dense NATIVE-u8 volume (8 GiB for a 2048³, NOT the
    // 34 GiB f32 slab), then gather patches from that flat array (a plain copy, no per-patch DCT decode, no
    // cache locks). This is far faster than streaming/re-decoding overlapping tiles per patch, and still
    // never widens u8→f32 for storage. (Streaming/out-of-core stays available via the archive's block cache
    // for volumes that don't fit in RAM — not needed here.)
    const bool prof0 = std::getenv("FENIX_INFER_PROFILE") != nullptr;
    auto clk0 = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double td = clk0();
    const bool fxvol_in = inpath.size() > 6 && inpath.substr(inpath.size() - 6) == ".fxvol";
    Volume<u8> vol_u8;
    Volume<f32> dense_vol;
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
            dense_vol = std::move(*v);
            d = dense_vol.dims();
        }
    } else {
        auto vol = load_volume(inpath);
        if (!vol) return std::unexpected(vol.error());
        dense_vol = std::move(*vol);
        d = dense_vol.dims();
    }
    if (prof0) fenix::log(LogLevel::info, "T decode-input: {:.1f}s", clk0() - td);
    const int tta_n = opt.tta <= 1 ? 1 : (opt.tta < 48 ? opt.tta : 48);
    fenix::log(LogLevel::info,
               "{}: input {}x{}x{} (ZYX), patch={} overlap={} tta={} src={}",
               name,
               d.z,
               d.y,
               d.x,
               opt.patch,
               opt.overlap,
               tta_n,
               u8_src ? "fxvol(dense u8)" : "dense f32");

    const bool prof = std::getenv("FENIX_INFER_PROFILE") != nullptr;
    auto clk = [] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    double ts = clk();

#ifdef FENIX_TRT
    // TensorRT engine (.plan/.engine): the bulk-inference fast path (1.53x over eager fp16,
    // measured — configs/gpu/rtx6000pro.toml). Engines are STATIC-shape and version/arch-locked;
    // the engine's own batch/patch override the CLI's (a mismatched patch would silently change
    // the tiling geometry, so the engine is authoritative).
    if (wpath.size() > 4 && (wpath.ends_with(".plan") || wpath.ends_with(".engine"))) {
        const auto dev0 = best_device();
        if (!dev0.is_cuda()) return fenix::err(Errc::unsupported, std::string(name) + ": trt engine needs CUDA");
        auto tnet = TrtNet::load(wpath);
        if (!tnet) return std::unexpected(tnet.error());
        if (opt.patch != tnet->patch() || opt.batch != static_cast<int>(tnet->batch()))
            fenix::log(LogLevel::info,
                       "{}: trt engine overrides patch {}->{} batch {}->{}",
                       name,
                       opt.patch,
                       tnet->patch(),
                       opt.batch,
                       tnet->batch());
        opt.patch = static_cast<int>(tnet->patch());
        opt.batch = static_cast<int>(tnet->batch());
        if (opt.tta > 1 || !opt.scales.empty() || !opt.rots.empty())
            fenix::log(LogLevel::info, "{}: trt engine with TTA — members run at engine batch", name);
        fenix::log(LogLevel::info,
                   "{}: trt engine loaded on {} (batch={} patch={} classes={})",
                   name,
                   dev0.str(),
                   tnet->batch(),
                   tnet->patch(),
                   tnet->classes());
        return run_predict_core(name, inpath, wpath, outpath, opt, *tnet, dev0, d, vol_u8, dense_vol, u8_src);
    }
#endif

    // TorchScript weights (.ts/.torchscript): the student export path — load the scripted module
    // and run it through the same predict machinery via the JitNet adapter (see run_predict_core).
    if (wpath.size() > 3 && (wpath.ends_with(".ts") || wpath.ends_with(".torchscript"))) {
        const auto dev0 = best_device();
        try {
            JitNet jnet{torch::jit::load(wpath, dev0)};
            jnet.m.eval();
            fenix::log(LogLevel::info, "{}: torchscript model loaded on {}", name, dev0.str());
            return run_predict_core(name, inpath, wpath, outpath, opt, jnet, dev0, d, vol_u8, dense_vol, u8_src);
        } catch (const std::exception& e) {
            return fenix::err(Errc::internal, std::string(name) + ": torchscript load failed: " + e.what());
        }
    }

    nets::ResEncUNet net(cfg);
    if (prof) {
        fenix::log(LogLevel::info, "T net-build: {:.1f}s", clk() - ts);
        ts = clk();
    }
    const auto dev = best_device();
    // net->to(dev) and load_fxweights/load_into are all torch calls that can throw (CUDA OOM moving a
    // multi-GB model to device, or a corrupt/truncated .fxweights hitting an internal torch assert during
    // from_blob/clone). This is the setup-side counterpart of the loop-body exception boundary in
    // infer.hpp — same reasoning: FENIX_ML builds compile WITH exceptions for the libtorch ABI, and an
    // uncaught exception here would std::terminate instead of returning fenix::err.
    std::vector<std::string> missing;
    try {
        net->to(dev);
        net->eval();
        if (prof) {
            torch::cuda::synchronize();
            fenix::log(LogLevel::info, "T net->to(dev): {:.1f}s", clk() - ts);
            ts = clk();
        }
    } catch (const std::exception& e) {
        return fenix::err(Errc::internal, std::string(name) + ": net->to(device) failed: " + e.what());
    }
    auto w = load_fxweights(wpath, dev);
    if (!w) return std::unexpected(w.error());
    if (prof) {
        fenix::log(LogLevel::info, "T load_fxweights: {:.1f}s", clk() - ts);
        ts = clk();
    }
    try {
        load_into(*net, *w, &missing);
        if (prof) {
            torch::cuda::synchronize();
            fenix::log(LogLevel::info, "T load_into(copy): {:.1f}s", clk() - ts);
            ts = clk();
        }
    } catch (const std::exception& e) {
        return fenix::err(Errc::internal, std::string(name) + ": load_into(weights) failed: " + e.what());
    }
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "{}: {} model params unmatched (e.g. {})", name, missing.size(), missing[0]);
        return fenix::err(Errc::decode_error, std::string(name) + ": weights/arch mismatch");
    }
    fenix::log(LogLevel::info, "{}: model loaded on {}", name, dev.str());

    return run_predict_core(name, inpath, wpath, outpath, opt, net, dev, d, vol_u8, dense_vol, u8_src);
}

// ---- torch-free entry points (declared in ml/ml_api.hpp) -------------------------------------------

// `fenix predict-surface <in> <weights.fxweights|.ts> <out> [patch] [overlap] [...] [model=<preset>]`
// model presets (introspected from the published checkpoints — docs/design/model-registry.md):
//   recto (default) | m7 | fibers2 | fibers4 | copy
Expected<int> run_predict_surface(std::span<const std::string_view> args) {
    nets::ResEncUNetConfig cfg;  // defaults = surface_recto_3dunet (task_decoders.surface, 2-class softmax)
    InferOptions opt;            // zscore, softmax channel 1
    for (const auto& a : args) {
        if (!a.starts_with("model=")) continue;
        const auto m = a.substr(6);
        if (m == "recto") {
        } else if (m == "m7")
            cfg = nets::ResEncUNetConfig::surface_m7();
        else if (m == "fibers2") {
            cfg = nets::ResEncUNetConfig::fibers(2);
        } else if (m == "fibers4") {
            cfg = nets::ResEncUNetConfig::fibers(4);
        } else if (m == "copy") {
            cfg = nets::ResEncUNetConfig::copy_displacement();
        } else
            return fenix::err(Errc::invalid_argument, "predict: unknown model preset '" + std::string(m) + "'");
    }
    return run_predict(args, "predict-surface", cfg, opt);
}

// `fenix predict-ink <in> <ink.fxweights> <out> [patch] [overlap]`

// `fenix predict-ink2d <stack.fxvol> <ink2d.fxweights> <out.jpg> [tile=256] [overlap=0.25]
//                      [dwin=62] [batch=2]` — production tiled inference for the r152+FPN
// 2D-ink model (ink_canonical_2um): the stack (D layers, H, W from render-layers) is tiled
// over H,W; each tile's depth is center-cropped/replicate-padded to dwin; sigmoid logits
// are Hann-blended into one 2D ink map written as a grayscale JPEG (prob*255).
Expected<int> run_predict_ink2d(std::span<const std::string_view> a) {
    if (a.size() < 3)
        return fenix::err(Errc::invalid_argument,
                          "usage: predict-ink2d <stack.fxvol> <ink2d.fxweights> <out.jpg> "
                          "[net=r152|r50] [tile=] [overlap=] [dwin=] [batch=]");
    std::string net_kind = "r152";
    for (std::size_t i = 3; i < a.size(); ++i)
        if (a[i].starts_with("net=")) net_kind = std::string(a[i].substr(4));
    const bool r50 = net_kind == "r50";
    if (!r50 && net_kind != "r152")
        return fenix::err(Errc::invalid_argument, "predict-ink2d: net must be r152|r50");
    // r152 (ink_canonical_2um): 256-tile, 62-layer window. r50 (resnet50_3um_01122024):
    // window_size=256, num_layers=18 per the HF config; NOTE its native resolution is 3um
    // (render layers with step = 3/2.4 = 1.25 on 2.4um-era volumes).
    long tile = 256, dwin = r50 ? 18 : 62, batch = r50 ? 8 : 2;
    double overlap = 0.25;
    for (std::size_t i = 3; i < a.size(); ++i) {
        const auto s2 = a[i];
        auto num = [&](std::string_view key, auto& v) {
            if (!s2.starts_with(key)) return false;
            const auto t = s2.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (s2.starts_with("net=") || num("tile=", tile) || num("dwin=", dwin) || num("batch=", batch) ||
            num("overlap=", overlap))
            continue;
        return fenix::err(Errc::invalid_argument, "predict-ink2d: unknown arg '" + std::string(s2) + "'");
    }

    const auto dev = best_device();
    auto w = load_fxweights(std::string(a[1]), dev);
    if (!w) return std::unexpected(w.error());
    nets::ResNet3DInk net152{nullptr};
    nets::ResNet3DInk50 net50{nullptr};
    std::function<torch::Tensor(torch::Tensor)> fwd;
    std::vector<std::string> missing;
    if (r50) {
        const bool with_norm = w->count("normalization.weight") > 0;
        net50 = nets::ResNet3DInk50(with_norm);
        net50->to(dev);
        net50->eval();
        load_into(*net50, *w, &missing);
        fwd = [&net50](torch::Tensor x) { return net50->forward(x); };
    } else {
        net152 = nets::ResNet3DInk(true);
        net152->to(dev);
        net152->eval();
        load_into(*net152, *w, &missing);
        fwd = [&net152](torch::Tensor x) { return net152->forward(x); };
    }
    if (!missing.empty()) {
        fenix::log(LogLevel::error, "predict-ink2d: {} params unmatched (e.g. {})", missing.size(), missing[0]);
        return fenix::err(Errc::decode_error, "predict-ink2d: weights/arch mismatch");
    }

    auto arch = codec::VolumeArchive::open(std::string(a[0]));
    if (!arch) return std::unexpected(arch.error());
    arch->reserve_cache(u64{4} << 30);
    const Extent3 d = arch->dims();
    const long D = d.z, H = static_cast<long>(d.y), W = static_cast<long>(d.x);
    // depth window: center-crop when the stack is deeper (render-layers pads z up), else
    // replicate the outer layers
    const long z0 = std::max<long>(0, (D - dwin) / 2);

    const long stride = std::max<long>(16, static_cast<long>(static_cast<double>(tile) * (1.0 - overlap)));
    std::vector<float> acc(static_cast<std::size_t>(H) * static_cast<std::size_t>(W), 0.0f);
    std::vector<float> wacc(acc.size(), 0.0f);
    std::vector<float> hann(static_cast<std::size_t>(tile) * static_cast<std::size_t>(tile));
    for (long y = 0; y < tile; ++y)
        for (long x = 0; x < tile; ++x) {
            const float wy = 0.5f - 0.5f * std::cos(2.0f * 3.14159265f * (static_cast<float>(y) + 0.5f) / tile);
            const float wx = 0.5f - 0.5f * std::cos(2.0f * 3.14159265f * (static_cast<float>(x) + 0.5f) / tile);
            hann[static_cast<std::size_t>(y * tile + x)] = wy * wx + 1e-4f;
        }

    std::vector<std::array<long, 2>> tiles;
    for (long ty = 0;; ty += stride) {
        const long y = std::min(ty, std::max<long>(0, H - tile));
        for (long tx = 0;; tx += stride) {
            const long x = std::min(tx, std::max<long>(0, W - tile));
            tiles.push_back({y, x});
            if (x >= W - tile || W <= tile) break;
        }
        if (y >= H - tile || H <= tile) break;
    }

    torch::NoGradGuard ng;
    std::vector<u8> tbuf(static_cast<std::size_t>(dwin) * static_cast<std::size_t>(tile) * static_cast<std::size_t>(tile));
    for (std::size_t t0 = 0; t0 < tiles.size(); t0 += static_cast<std::size_t>(batch)) {
        const std::size_t nb = std::min<std::size_t>(static_cast<std::size_t>(batch), tiles.size() - t0);
        auto x = torch::empty({static_cast<long>(nb), 1, dwin, tile, tile}, torch::kFloat32);
        for (std::size_t b = 0; b < nb; ++b) {
            const auto [ty, tx] = tiles[t0 + b];
            for (long zz = 0; zz < dwin; ++zz) {
                const long src_z = std::clamp<long>(z0 + zz, 0, D - 1);
                if (auto g = arch->gather_box_u8(0, src_z, ty, tx, 1, tile, tile,
                                                 tbuf.data() + static_cast<std::size_t>(zz) * tile * tile);
                    !g)
                    return std::unexpected(g.error());
            }
            auto acc_t = x[static_cast<long>(b)][0];
            for (long zz = 0; zz < dwin; ++zz)
                for (long yy = 0; yy < tile; ++yy)
                    for (long xx = 0; xx < tile; ++xx)
                        acc_t[zz][yy][xx] =
                            static_cast<float>(tbuf[static_cast<std::size_t>((zz * tile + yy) * tile + xx)]) / 255.0f;
        }
        auto y = torch::sigmoid(fwd(x.to(dev)));  // (nb, 1, H', W') — H' is input/4 for both nets
        if (y.dim() == 3) y = y.unsqueeze(1);
        // resize probs back to tile size BEFORE blending (mirrors upstream inference.py —
        // pasting the low-res logit map unscaled quarter-fills the tile)
        if (y.size(2) != tile || y.size(3) != tile)
            y = torch::nn::functional::interpolate(
                y, torch::nn::functional::InterpolateFuncOptions()
                       .size(std::vector<int64_t>{tile, tile}).mode(torch::kBilinear).align_corners(false));
        y = y.to(torch::kCPU).contiguous();
        for (std::size_t b = 0; b < nb; ++b) {
            const auto [ty, tx] = tiles[t0 + b];
            auto yb = y[static_cast<long>(b)][0];
            const float* py = yb.data_ptr<float>();
            for (long yy = 0; yy < tile; ++yy)
                for (long xx = 0; xx < tile; ++xx) {
                    const std::size_t gi = static_cast<std::size_t>((ty + yy) * W + (tx + xx));
                    const float hw = hann[static_cast<std::size_t>(yy * tile + xx)];
                    acc[gi] += py[yy * tile + xx] * hw;
                    wacc[gi] += hw;
                }
        }
        if ((t0 / static_cast<std::size_t>(batch)) % 8 == 0)
            fenix::log(LogLevel::info, "predict-ink2d: {}/{} tiles", t0 + nb, tiles.size());
    }

    io::Image img;
    img.w = static_cast<int>(W);
    img.h = static_cast<int>(H);
    img.comps = 1;
    img.px.resize(acc.size());
    for (std::size_t i = 0; i < acc.size(); ++i) {
        const float v = wacc[i] > 0 ? acc[i] / wacc[i] : 0.0f;
        img.px[i] = static_cast<u8>(std::clamp(v, 0.0f, 1.0f) * 255.0f + 0.5f);
    }
    if (auto wj = io::write_jpeg(std::string(a[2]), img, 95); !wj) return std::unexpected(wj.error());
    fenix::log(LogLevel::info, "predict-ink2d: {} tiles -> {} ({}x{})", tiles.size(), a[2], W, H);
    return 0;
}

Expected<int> run_predict_ink(std::span<const std::string_view> args) {
    nets::ResEncUNetConfig cfg;
    cfg.task = "ink";
    cfg.num_classes = 1;
    cfg.task_head = true;            // shared_decoder + task_heads.ink (single 1x1 conv)
    cfg.squeeze_excitation = false;  // ink encoder has plain residual blocks (no scSE)
    InferOptions opt;
    opt.sigmoid = true;           // 1-channel logit -> sigmoid
    opt.norm = Norm::pct_minmax;  // percentile (0.5/99.5) min-max
    return run_predict(args, "predict-ink", cfg, opt);
}

// In-process per-window surface inference for the STREAMED TRACER: one resident u8
// window -> u8 sheet probability, same dims. The net is built + loaded ONCE per weights
// path (static cache — the fetch fn calls this per window). Patch 128 (windows are
// ~384^3; 256 wastes halo), fp16, zscore, softmax channel 1.
Expected<Volume<u8>> predict_surface_window(VolumeView<const u8> ct, const std::string& weights_path) {
    struct CachedNet {
        std::string path;
        nets::ResEncUNet net{nullptr};
    };
    static CachedNet cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk(mu);
    const auto dev = best_device();
    if (cache.path != weights_path) {
        nets::ResEncUNet net(nets::ResEncUNetConfig{});
        net->to(dev);
        net->eval();
        auto w = load_fxweights(weights_path, dev);
        if (!w) return std::unexpected(w.error());
        std::vector<std::string> missing;
        load_into(*net, *w, &missing);
        if (!missing.empty()) return fenix::err(Errc::decode_error, "predict_surface_window: weights/arch mismatch");
        cache.net = net;
        cache.path = weights_path;
    }
    InferOptions opt;
    opt.patch = 128;
    opt.overlap = 0.25;
    torch::NoGradGuard ng;
    auto prob = predict_surface_filled(
        ct.dims(),
        [ct](s64 z0, s64 y0, s64 x0, int P, float* out) {
            parallel_for(0, P, [&](s64 z) {
                for (int y = 0; y < P; ++y)
                    for (int x = 0; x < P; ++x)
                        out[(static_cast<std::size_t>(z) * P + y) * P + x] =
                            static_cast<f32>(ct.at_clamped(z0 + z, y0 + y, x0 + x));
            });
        },
        cache.net,
        dev,
        opt);
    if (!prob) return std::unexpected(prob.error());
    Volume<u8> out(prob->dims());
    auto ov = out.view();
    auto pv = prob->view();
    parallel_for(0, out.dims().z, [&](s64 z) {
        for (s64 y = 0; y < out.dims().y; ++y)
            for (s64 x = 0; x < out.dims().x; ++x)
                ov(z, y, x) = static_cast<u8>(std::clamp(pv(z, y, x), 0.0f, 1.0f) * 255.0f + 0.5f);
    });
    return out;
}

// `fenix predict-scroll <zarr-root|url> <level> <surface.fxweights> <out_pred.fxvol> [ct=<raw_ct.fxvol>]
//                        [tta=8] [batch=2] [patch=256] [overlap=0.5] [region=512] [q=32] [ctq=2] [commit=64]`
//
// WHOLE-SCROLL surface prediction, OUT-OF-CORE + RESUMABLE, across ALL visible GPUs in ONE process.
// Mirrors io::export_scroll's producer/consumer + coverage-resume skeleton, but inserts a multi-GPU
// COMPUTE stage between the CT gather and the archive write:
//
//   [N GPU workers]  each: claim a region tile → gather its CT from the zarr (with halo) → run
//                    octahedral-TTA sliding-window inference on ITS OWN device (one model replica per
//                    GPU, c10::cuda::CUDAGuard) → push (ct_block, pred_block) to the writer.
//   [single writer]  pops results and write_chunk()s the prediction into out_pred.fxvol AND (if ct=)
//                    the raw CT into raw_ct.fxvol at the tile's chunk coords; batched crash-safe commit.
//
// The .fxvol archive is single-writer (flock/coverage), so exactly ONE thread ever touches each archive
// (the writer) — the GPUs never do IO on the archives. CT is fetched from S3 ONCE and lands in BOTH the
// raw-CT archive and (as the model input) the prediction. Resume: a region whose prediction tiles are
// already present (Real/Zero) is skipped; a crash mid-region leaves those tiles ABSENT so a rerun redoes
// only the in-flight regions. Air-skip: regions with no non-zero coarse-pyramid voxel are left ABSENT.
Expected<int> run_predict_scroll(std::span<const std::string_view> args) {
    if (args.size() < 4) {
        fenix::log(LogLevel::error,
                   "usage: predict-scroll <zarr-root|url> <level> <surface.fxweights> <out_pred.fxvol> "
                   "[ct=<raw_ct.fxvol>] [tta=8] [batch=2] [patch=256] [overlap=0.5] [region=512] [q=32] "
                   "[ctq=2] [commit=64]");
        return fenix::err(Errc::invalid_argument, "missing args");
    }
    ::setenv("KMP_BLOCKTIME", "0", 0);
    ::setenv("OMP_WAIT_POLICY", "passive", 0);
    std::string root(args[0]);
    while (!root.empty() && root.back() == '/') root.pop_back();
    const std::string level(args[1]);
    const std::string lroot = root + "/" + level;
    const std::string wpath(args[2]);
    const std::string out_pred(args[3]);
    std::string ct_path;
    int tta = 8, batch = 2, patch = 256;
    f64 overlap = 0.5;
    s64 R = 512, commit_every = 64;
    f32 predq = 32.0f, ctq = 2.0f;
    auto kv = [&](std::string_view a, std::string_view k) -> std::optional<std::string_view> {
        if (a.size() > k.size() + 1 && a.substr(0, k.size()) == k && a[k.size()] == '=') return a.substr(k.size() + 1);
        return std::nullopt;
    };
    for (usize i = 4; i < args.size(); ++i) {
        const auto a = args[i];
        if (auto v = kv(a, "ct")) ct_path = std::string(*v);
        else if (auto v = kv(a, "tta")) std::from_chars(v->data(), v->data() + v->size(), tta);
        else if (auto v = kv(a, "batch")) std::from_chars(v->data(), v->data() + v->size(), batch);
        else if (auto v = kv(a, "patch")) std::from_chars(v->data(), v->data() + v->size(), patch);
        else if (auto v = kv(a, "overlap")) std::from_chars(v->data(), v->data() + v->size(), overlap);
        else if (auto v = kv(a, "region")) std::from_chars(v->data(), v->data() + v->size(), R);
        else if (auto v = kv(a, "commit")) std::from_chars(v->data(), v->data() + v->size(), commit_every);
        else if (auto v = kv(a, "q")) std::from_chars(v->data(), v->data() + v->size(), predq);
        else if (auto v = kv(a, "ctq")) std::from_chars(v->data(), v->data() + v->size(), ctq);
        else return fenix::err(Errc::invalid_argument, "predict-scroll: unknown arg '" + std::string(a) + "'");
    }
    if (patch % 64 != 0) return fenix::err(Errc::invalid_argument, "patch must be divisible by 64");
    const s64 T = codec::fxvol_chunk_side;  // 64
    R = std::max<s64>(T, (R / T) * T);
    commit_every = std::max<s64>(1, commit_every);
    auto ndiv = [](s64 n, s64 d) { return (n + d - 1) / d; };

    // Zarr geometry.
    auto meta = io::read_zarray(lroot);
    if (!meta) return std::unexpected(meta.error());
    const Extent3 shape = meta->shape;
    if (io::detail::dtype_size(meta->dtype) != 1)
        return fenix::err(Errc::unsupported, "predict-scroll: only u8 (|u1) zarr sources supported, got " + meta->dtype);

    // GPU model replicas — one per visible device (falls back to a single CPU replica if no CUDA).
    const int ndev = torch::cuda::is_available() ? static_cast<int>(torch::cuda::device_count()) : 1;
    std::vector<nets::ResEncUNet> replicas;
    std::vector<torch::Device> devs;
    replicas.reserve(static_cast<usize>(ndev));
    for (int g = 0; g < ndev; ++g) {
        torch::Device dev = torch::cuda::is_available() ? torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(g))
                                                        : torch::Device(torch::kCPU);
        nets::ResEncUNet net(nets::ResEncUNetConfig{});
        net->to(dev);
        net->eval();
        auto w = load_fxweights(wpath, dev);
        if (!w) return std::unexpected(w.error());
        std::vector<std::string> missing;
        load_into(*net, *w, &missing);
        if (!missing.empty()) return fenix::err(Errc::decode_error, "predict-scroll: weights/arch mismatch on device " + std::to_string(g));
        replicas.push_back(net);
        devs.push_back(dev);
    }

    // Coarse occupancy map for air-skip (this is a masked volume — most of the box is air). Load the
    // coarsest available pyramid level; skip a region only if NO coarse voxel in its footprint is non-zero.
    Volume<u8> occ;
    s64 occ_scale = 1;
    s64 level_int = 0;
    std::from_chars(level.data(), level.data() + level.size(), level_int);
    for (s64 k = 5; k >= 1; --k) {
        auto m = io::read_zarray(root + "/" + std::to_string(level_int + k));
        if (!m) continue;
        auto o = io::read_zarr_region<u8>(root + "/" + std::to_string(level_int + k), {0, 0, 0}, m->shape);
        if (!o) continue;
        occ = std::move(*o);
        occ_scale = static_cast<s64>(1) << static_cast<u32>(k);
        break;
    }
    auto occupied = [&](Index3 org, Extent3 ext) -> bool {
        if (occ.dims().z == 0) return true;
        const Extent3 os = occ.dims();
        auto ov = occ.view();
        const s64 z0 = std::max<s64>(0, org.z / occ_scale - 1), z1 = std::min(os.z, (org.z + ext.z + occ_scale - 1) / occ_scale + 1);
        const s64 y0 = std::max<s64>(0, org.y / occ_scale - 1), y1 = std::min(os.y, (org.y + ext.y + occ_scale - 1) / occ_scale + 1);
        const s64 x0 = std::max<s64>(0, org.x / occ_scale - 1), x1 = std::min(os.x, (org.x + ext.x + occ_scale - 1) / occ_scale + 1);
        for (s64 z = z0; z < z1; ++z)
            for (s64 y = y0; y < y1; ++y)
                for (s64 x = x0; x < x1; ++x)
                    if (ov(z, y, x) != 0) return true;
        return false;
    };

    // Create/open the output archives at FULL scroll dims (sparse page table + coverage sentinels).
    // Resume: open an EXISTING archive writable (COW preserves the committed snapshot); create() O_TRUNCs.
    const bool resume = std::filesystem::exists(out_pred);
    auto pred_a = resume ? codec::VolumeArchive::open(out_pred, true)
                         : codec::VolumeArchive::create(out_pred, shape, codec::DctParams{.q = predq});
    if (!pred_a) return std::unexpected(pred_a.error());
    if (resume && !(pred_a->dims() == shape)) return fenix::err(Errc::invalid_argument, "predict-scroll resume: pred archive dims != zarr shape");
    std::optional<codec::VolumeArchive> ct_a;
    if (!ct_path.empty()) {
        const bool ct_resume = std::filesystem::exists(ct_path);
        auto c = ct_resume ? codec::VolumeArchive::open(ct_path, true)
                           : codec::VolumeArchive::create(ct_path, shape, codec::DctParams{.q = ctq});
        if (!c) return std::unexpected(c.error());
        if (ct_resume && !(c->dims() == shape)) return fenix::err(Errc::invalid_argument, "predict-scroll resume: ct archive dims != zarr shape");
        ct_a = std::move(*c);
    }

    // Region grid.
    const s64 nrz = ndiv(shape.z, R), nry = ndiv(shape.y, R), nrx = ndiv(shape.x, R);
    const s64 total = nrz * nry * nrx;

    // Worklist: occupied, not-yet-done regions (resume-skip on the PREDICTION archive's coverage).
    struct Work { Index3 org; Extent3 ext; };
    std::vector<Work> work;
    s64 skipped = 0, skipped_air = 0;
    for (s64 rz = 0; rz < nrz; ++rz)
        for (s64 ry = 0; ry < nry; ++ry)
            for (s64 rx = 0; rx < nrx; ++rx) {
                const Index3 org{rz * R, ry * R, rx * R};
                const Extent3 ext{std::min(R, shape.z - org.z), std::min(R, shape.y - org.y), std::min(R, shape.x - org.x)};
                const ChunkCoord ft{org.z / T, org.y / T, org.x / T};
                if (pred_a->coverage(0, ft) != codec::Coverage::Absent) { ++skipped; continue; }
                if (!occupied(org, ext)) { ++skipped_air; continue; }
                work.push_back({org, ext});
            }
    const s64 nwork = static_cast<s64>(work.size());
    fenix::log(LogLevel::info,
               "predict-scroll {} L{} ({}x{}x{}) -> {}{}  {} GPU(s) tta={} batch={} patch={} region={}  "
               "{} regions to run ({} air-skip, {} resume-skip of {})  {}",
               root, level, shape.z, shape.y, shape.x, out_pred, ct_path.empty() ? "" : (" + " + ct_path).c_str(),
               ndev, tta, batch, patch, R, nwork, skipped_air, skipped, total, resume ? "RESUME" : "fresh");

    // Halo so a region's border patches see context from the neighbours (avoids seam artifacts): the model
    // sees [org-halo, org+ext+halo], predicts, and only the core [org, org+ext] is written. halo = patch/2
    // (one patch of context each side) rounded to the tile grid.
    const s64 halo = ((patch / 2 + T - 1) / T) * T;

    // Producer/consumer with a GPU-compute middle. Each GPU worker owns device g and pulls regions off the
    // shared cursor; it gathers CT (with halo, clamped to the volume), runs TTA inference on device g, and
    // pushes the finished (core CT block, core pred block) to the writer queue. The single writer thread
    // (this thread) drains the queue into the archives.
    struct Result { s64 idx; Volume<u8> ct_core; Volume<u8> pred_core; };
    std::mutex mtx;
    std::condition_variable cv_push, cv_pop;
    std::deque<Result> queue;
    const usize qcap = static_cast<usize>(ndev) + 1;
    std::atomic<s64> cursor{0};
    int workers_live = ndev;
    bool cancel = false;
    Error worker_err;
    bool has_err = false;

    InferOptions iopt;
    iopt.patch = patch;
    iopt.overlap = overlap;
    iopt.tta = tta;
    iopt.batch = std::max(1, batch);

    auto worker = [&](int g) {
        // No explicit CUDAGuard: each replica already lives on devs[g] and predict_surface_filled moves
        // every patch to that device, so libtorch dispatches all ops on the correct device per worker.
        // (An OptionalCUDAGuard would pull raw cudart symbols the rest of fenix deliberately doesn't link.)
        torch::NoGradGuard ng;
        for (;;) {
            const s64 i = cursor.fetch_add(1);
            if (i >= nwork) break;
            { std::unique_lock lk(mtx); if (cancel) break; }
            const Work w = work[static_cast<usize>(i)];
            // Halo-expanded gather box, clamped to the volume.
            const Index3 gorg{std::max<s64>(0, w.org.z - halo), std::max<s64>(0, w.org.y - halo), std::max<s64>(0, w.org.x - halo)};
            const Index3 gend{std::min(shape.z, w.org.z + w.ext.z + halo), std::min(shape.y, w.org.y + w.ext.y + halo), std::min(shape.x, w.org.x + w.ext.x + halo)};
            const Extent3 gext{gend.z - gorg.z, gend.y - gorg.y, gend.x - gorg.x};
            // Gather CT (retry through transient S3 blips — a multi-day run must not die on one GET).
            Expected<Volume<u8>> ctv = io::read_zarr_region<u8>(lroot, gorg, gext);
            for (int attempt = 1; !ctv && attempt <= 20; ++attempt) {
                { std::unique_lock lk(mtx); if (cancel) break; }
                const int backoff = std::min(30, 1 << std::min(attempt, 5));
                fenix::log(LogLevel::warn, "predict-scroll region z{} y{} x{} CT read failed: {} — retry {}/20 in {}s",
                           gorg.z, gorg.y, gorg.x, ctv.error().message, attempt, backoff);
                std::this_thread::sleep_for(std::chrono::seconds(backoff));
                ctv = io::read_zarr_region<u8>(lroot, gorg, gext);
            }
            if (!ctv) { std::unique_lock lk(mtx); if (!has_err) { worker_err = ctv.error(); has_err = true; } cancel = true; cv_pop.notify_all(); break; }
            // Octahedral-TTA sliding-window inference on THIS device, gathering each patch straight from the
            // dense u8 CT (widened to f32 per patch, edge-clamped) — the same filler predict_surface_window
            // uses. opt.tta drives the octahedral ensemble; the multi-scale wrapper isn't used here.
            auto ctview = ctv->view();
            Expected<Volume<f32>> probf = predict_surface_filled(
                ctview.dims(),
                [ctview](s64 z0, s64 y0, s64 x0, int P, float* out) {
                    parallel_for(0, P, [&](s64 z) {
                        for (int y = 0; y < P; ++y)
                            for (int x = 0; x < P; ++x)
                                out[(static_cast<std::size_t>(z) * P + y) * P + x] =
                                    static_cast<f32>(ctview.at_clamped(z0 + z, y0 + y, x0 + x));
                    });
                },
                replicas[static_cast<usize>(g)], devs[static_cast<usize>(g)], iopt);
            if (!probf) { std::unique_lock lk(mtx); if (!has_err) { worker_err = probf.error(); has_err = true; } cancel = true; cv_pop.notify_all(); break; }
            // Crop the CORE [w.org, w.org+ext] out of the halo-expanded blocks, to u8.
            const Index3 coff{w.org.z - gorg.z, w.org.y - gorg.y, w.org.x - gorg.x};
            Volume<u8> pred_core(w.ext), ct_core(w.ext);
            auto pf = probf->view(); auto cin = ctv->view();
            auto pc = pred_core.view(); auto cc = ct_core.view();
            for (s64 z = 0; z < w.ext.z; ++z)
                for (s64 y = 0; y < w.ext.y; ++y)
                    for (s64 x = 0; x < w.ext.x; ++x) {
                        pc(z, y, x) = static_cast<u8>(std::clamp(pf(z + coff.z, y + coff.y, x + coff.x), 0.0f, 1.0f) * 255.0f + 0.5f);
                        cc(z, y, x) = cin(z + coff.z, y + coff.y, x + coff.x);
                    }
            std::unique_lock lk(mtx);
            cv_push.wait(lk, [&] { return queue.size() < qcap || cancel; });
            if (cancel) break;
            queue.push_back({i, std::move(ct_core), std::move(pred_core)});
            cv_pop.notify_one();
        }
        std::unique_lock lk(mtx);
        if (--workers_live == 0) cv_pop.notify_all();
    };

    std::vector<std::thread> pool;
    for (int g = 0; g < ndev; ++g) pool.emplace_back(worker, g);

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto t_last = t0;
    s64 done = 0, since_commit = 0;
    u64 real_tiles = 0, zero_tiles = 0;
    Expected<void> write_err;
    std::vector<u8> blk(static_cast<usize>(T * T * T));
    auto write_core = [&](codec::VolumeArchive& a, const Volume<u8>& core, Index3 org, Extent3 ext) -> bool {
        auto cv = core.view();
        for (s64 tz = 0; tz < ndiv(ext.z, T); ++tz)
            for (s64 ty = 0; ty < ndiv(ext.y, T); ++ty)
                for (s64 tx = 0; tx < ndiv(ext.x, T); ++tx) {
                    for (s64 z = 0; z < T; ++z)
                        for (s64 y = 0; y < T; ++y)
                            for (s64 x = 0; x < T; ++x)
                                blk[static_cast<usize>((z * T + y) * T + x)] =
                                    cv(std::min(tz * T + z, ext.z - 1), std::min(ty * T + y, ext.y - 1), std::min(tx * T + x, ext.x - 1));
                    const ChunkCoord tc{org.z / T + tz, org.y / T + ty, org.x / T + tx};
                    if (auto w = a.write_chunk(0, tc, std::span<const u8>(blk)); !w) { write_err = std::unexpected(w.error()); return false; }
                }
        return true;
    };
    for (;;) {
        Result r;
        {
            std::unique_lock lk(mtx);
            cv_pop.wait(lk, [&] { return !queue.empty() || workers_live == 0; });
            if (queue.empty()) break;
            r = std::move(queue.front());
            queue.pop_front();
            cv_push.notify_one();
        }
        if (const auto nowt = clk::now(); std::chrono::duration<f64>(nowt - t_last).count() >= 15.0 || done == 0) {
            t_last = nowt;
            const f64 el = std::chrono::duration<f64>(nowt - t0).count();
            const f64 frac = static_cast<f64>(done) / static_cast<f64>(nwork ? nwork : 1);
            fenix::log(LogLevel::info, "  predict-scroll {}/{} regions ({:.1f}%) {:.0f}s ETA {:.0f}s | real {} zero {} tiles | {:.1f} MiB",
                       done, nwork, 100.0 * frac, el, frac > 0 ? el * (1.0 - frac) / frac : 0.0, real_tiles, zero_tiles,
                       static_cast<f64>(pred_a->committed_size()) / (1024.0 * 1024.0));
        }
        const Index3 org = work[static_cast<usize>(r.idx)].org;
        const Extent3 ext = work[static_cast<usize>(r.idx)].ext;
        bool ok = write_core(*pred_a, r.pred_core, org, ext);
        if (ok) for (s64 tz = 0; tz < ndiv(ext.z, T); ++tz) for (s64 ty = 0; ty < ndiv(ext.y, T); ++ty) for (s64 tx = 0; tx < ndiv(ext.x, T); ++tx)
            (pred_a->coverage(0, {org.z / T + tz, org.y / T + ty, org.x / T + tx}) == codec::Coverage::Real ? real_tiles : zero_tiles)++;
        if (ok && ct_a) ok = write_core(*ct_a, r.ct_core, org, ext);
        if (ok && ++since_commit >= commit_every) {
            if (auto c = pred_a->commit(); !c) { write_err = std::unexpected(c.error()); ok = false; }
            if (ok && ct_a) { if (auto c = ct_a->commit(); !c) { write_err = std::unexpected(c.error()); ok = false; } }
            since_commit = 0;
        }
        ++done;
        if (!ok) { std::unique_lock lk(mtx); cancel = true; cv_push.notify_all(); break; }
    }
    for (auto& th : pool) th.join();
    if (has_err) return std::unexpected(worker_err);
    if (!write_err) return std::unexpected(write_err.error());
    if (auto c = pred_a->close(); !c) return std::unexpected(c.error());
    if (ct_a) { if (auto c = ct_a->close(); !c) return std::unexpected(c.error()); }
    fenix::log(LogLevel::info, "predict-scroll done: {} regions run ({} air-skip, {} resume-skip), real {} / zero {} tiles, {:.1f} MiB pred",
               nwork, skipped_air, skipped, real_tiles, zero_tiles, static_cast<f64>(pred_a->committed_size()) / (1024.0 * 1024.0));
    return 0;
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
