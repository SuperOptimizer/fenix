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

#include <c10/cuda/CUDAFunctions.h>
#endif
#include "io/cached_volume.hpp"
#include "io/jpeg.hpp"
#include "ml/aoti_net.hpp"
#include "ml/global_accum.hpp"
#include "ml/nets/resnet3d.hpp"
#include "ml/torch_env.hpp"
#include "ml/weights.hpp"

#include <torch/script.h>  // torch::jit::load — the .ts student path

#include <charconv>
#include <chrono>
#include <filesystem>
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
    // Idle OMP workers SLEEP at barriers rather than busy-spin through the GPU forward / S3 fetch waits
    // (libomp spins for KMP_BLOCKTIME=200ms by default, pinning cores at ~0% useful work while the CPU
    // prep thread waits on the GPU). An explicit user env still wins. Matches predict-scroll/export-scroll.
    ::setenv("KMP_BLOCKTIME", "0", 0);
    ::setenv("OMP_WAIT_POLICY", "passive", 0);
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

    // AOTInductor package (.pt2): torch.compile's fused kernels in the C++ path (export_aoti.py;
    // measured 1.8x over eager on MI300X, 1.3x on 5060 Ti — the win is op fusion, convs stay
    // MIOpen/cuDNN). Packages are STATIC-shape and torch-version/arch-locked; the package's own
    // batch/patch override the CLI's (same authority rule as the TRT engine above).
    if (wpath.size() > 4 && wpath.ends_with(".pt2")) {
        const auto dev0 = best_device();
        auto anet = AotiNet::load(wpath, dev0);
        if (!anet) return std::unexpected(anet.error());
        if (opt.patch != anet->patch() || opt.batch != static_cast<int>(anet->batch()))
            fenix::log(LogLevel::info,
                       "{}: aoti package overrides patch {}->{} batch {}->{}",
                       name,
                       opt.patch,
                       anet->patch(),
                       opt.batch,
                       anet->batch());
        opt.patch = static_cast<int>(anet->patch());
        opt.batch = static_cast<int>(anet->batch());
        fenix::log(LogLevel::info,
                   "{}: aoti package loaded on {} (batch={} patch={} classes={})",
                   name,
                   dev0.str(),
                   anet->batch(),
                   anet->patch(),
                   anet->classes());
        return run_predict_core(name, inpath, wpath, outpath, opt, *anet, dev0, d, vol_u8, dense_vol, u8_src);
    }

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
    if (!r50 && net_kind != "r152") return fenix::err(Errc::invalid_argument, "predict-ink2d: net must be r152|r50");
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
    std::vector<u8> tbuf(static_cast<std::size_t>(dwin) * static_cast<std::size_t>(tile) *
                         static_cast<std::size_t>(tile));
    for (std::size_t t0 = 0; t0 < tiles.size(); t0 += static_cast<std::size_t>(batch)) {
        const std::size_t nb = std::min<std::size_t>(static_cast<std::size_t>(batch), tiles.size() - t0);
        auto x = torch::empty({static_cast<long>(nb), 1, dwin, tile, tile}, torch::kFloat32);
        for (std::size_t b = 0; b < nb; ++b) {
            const auto [ty, tx] = tiles[t0 + b];
            for (long zz = 0; zz < dwin; ++zz) {
                const long src_z = std::clamp<long>(z0 + zz, 0, D - 1);
                if (auto g = arch->gather_box_u8(
                        0, src_z, ty, tx, 1, tile, tile, tbuf.data() + static_cast<std::size_t>(zz) * tile * tile);
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
            y = torch::nn::functional::interpolate(y,
                                                   torch::nn::functional::InterpolateFuncOptions()
                                                       .size(std::vector<int64_t>{tile, tile})
                                                       .mode(torch::kBilinear)
                                                       .align_corners(false));
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

// `fenix predict-scroll <zarr-root|url|ct.fxvol> <level> <surface.fxweights> <out_pred.fxvol> [ct=<raw_ct.fxvol>]
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
// SOURCE: args[0] is either an OME-Zarr root/URL (fetched from S3/http/local zarr) OR a LOCAL .fxvol CT
// archive (detected by the .fxvol suffix + existing regular file). The local-.fxvol path decodes each halo
// region straight from disk instead of re-fetching TBs from S3 — the natural input when the CT was already
// export-scroll'd. In local mode <level> is ignored, occupancy comes from the archive's own coarse LOD, and
// each GPU worker opens its own read-only archive handle. It predicts at the archive's NATIVE grid (no
// resampling here; any up/downscale to a training grid happens later, at train-feed time).
//
// The .fxvol archive is single-writer (flock/coverage), so exactly ONE thread ever touches each OUTPUT
// archive (the writer) — the GPUs never do IO on the output archives. CT is fetched/decoded ONCE and lands
// in BOTH the raw-CT archive and (as the model input) the prediction. Resume: a region whose prediction tiles are
// already present (Real/Zero) is skipped; a crash mid-region leaves those tiles ABSENT so a rerun redoes
// only the in-flight regions. Air-skip: regions with no non-zero coarse-pyramid voxel are left ABSENT.
Expected<int> run_predict_scroll(std::span<const std::string_view> args) {
    if (args.size() < 4) {
        fenix::log(LogLevel::error,
                   "usage: predict-scroll <zarr-root|url|ct.fxvol> <level> <surface.fxweights> <out_pred.fxvol> "
                   "[ct=<raw_ct.fxvol>] [tta=8] [batch=2] [patch=256] [overlap=0.5] [mode=global|region] "
                   "[region=512] [halo=128] [ahead=4] "
                   "[q=32] [ctq=2] [commit=64] [bbox=z0,y0,x0,D,H,W] [gpuworkers=1]");
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
    s64 halo_arg = -1;    // -1 = default (patch/2)
    s64 gpu_workers = 1;  // model replicas (worker threads) PER device — >1 overlaps one region's
                          // gather/encode with another's GPU compute (big-VRAM cards: MI300X)
    std::string bbox_str;
    std::string mode = "global";  // global = zero-waste grid (default); region = the legacy halo grid
    s64 ahead_rows = 4;           // global mode: patch-row lookahead (bounds the live accumulator band)
    auto kv = [&](std::string_view a, std::string_view k) -> std::optional<std::string_view> {
        if (a.size() > k.size() + 1 && a.substr(0, k.size()) == k && a[k.size()] == '=') return a.substr(k.size() + 1);
        return std::nullopt;
    };
    for (usize i = 4; i < args.size(); ++i) {
        const auto a = args[i];
        if (auto v = kv(a, "ct"))
            ct_path = std::string(*v);
        else if (auto v = kv(a, "tta"))
            std::from_chars(v->data(), v->data() + v->size(), tta);
        else if (auto v = kv(a, "batch"))
            std::from_chars(v->data(), v->data() + v->size(), batch);
        else if (auto v = kv(a, "patch"))
            std::from_chars(v->data(), v->data() + v->size(), patch);
        else if (auto v = kv(a, "overlap"))
            std::from_chars(v->data(), v->data() + v->size(), overlap);
        else if (auto v = kv(a, "region"))
            std::from_chars(v->data(), v->data() + v->size(), R);
        else if (auto v = kv(a, "commit"))
            std::from_chars(v->data(), v->data() + v->size(), commit_every);
        else if (auto v = kv(a, "q"))
            std::from_chars(v->data(), v->data() + v->size(), predq);
        else if (auto v = kv(a, "ctq"))
            std::from_chars(v->data(), v->data() + v->size(), ctq);
        else if (auto v = kv(a, "halo"))
            std::from_chars(v->data(), v->data() + v->size(), halo_arg);
        else if (auto v = kv(a, "bbox"))
            bbox_str = std::string(*v);
        else if (auto v = kv(a, "gpuworkers"))
            std::from_chars(v->data(), v->data() + v->size(), gpu_workers);
        else if (auto v = kv(a, "mode"))
            mode = std::string(*v);
        else if (auto v = kv(a, "ahead"))
            std::from_chars(v->data(), v->data() + v->size(), ahead_rows);
        else
            return fenix::err(Errc::invalid_argument, "predict-scroll: unknown arg '" + std::string(a) + "'");
    }
    gpu_workers = std::clamp<s64>(gpu_workers, 1, 8);
    // Optional sub-box: bbox=z0,y0,x0,D,H,W limits the run to that region of the (full-shape) archive.
    // The output archives stay full-scroll dims; only the covered sub-box is populated (rest stays ABSENT).
    Index3 bb_org{0, 0, 0};
    Extent3 bb_ext{-1, -1, -1};  // -1 = whole volume
    if (!bbox_str.empty()) {
        s64 v[6] = {0, 0, 0, -1, -1, -1};
        const char* p = bbox_str.c_str();
        const char* end = p + bbox_str.size();
        for (int i = 0; i < 6 && p < end; ++i) {
            auto r = std::from_chars(p, end, v[i]);
            if (r.ec != std::errc{})
                return fenix::err(Errc::invalid_argument, "predict-scroll: bad bbox (want z0,y0,x0,D,H,W)");
            p = r.ptr;
            if (p < end && *p == ',') ++p;
        }
        bb_org = {v[0], v[1], v[2]};
        bb_ext = {v[3], v[4], v[5]};
    }
    if (patch % 64 != 0) return fenix::err(Errc::invalid_argument, "patch must be divisible by 64");
    if (mode != "global" && mode != "region")
        return fenix::err(Errc::invalid_argument, "predict-scroll: mode must be global or region");
    ahead_rows = std::clamp<s64>(ahead_rows, 1, 64);
    const s64 T = codec::fxvol_chunk_side;  // 64
    R = std::max<s64>(T, (R / T) * T);
    commit_every = std::max<s64>(1, commit_every);
    auto ndiv = [](s64 n, s64 d) { return (n + d - 1) / d; };

    // Source: a remote/local OME-Zarr root (default) OR a LOCAL .fxvol archive. The latter avoids
    // re-fetching TBs from S3 when the CT is already on disk (e.g. a previously export-scroll'd scroll).
    // Detect a local .fxvol by suffix + existence as a regular file; anything else stays the zarr path.
    const bool local_src = root.size() > 6 && root.substr(root.size() - 6) == ".fxvol" &&
                           std::filesystem::exists(root) && std::filesystem::is_regular_file(root);

    // Geometry (+ later: occupancy). Zarr → read_zarray; local .fxvol → the archive's own dims (u8-native).
    Extent3 shape{};
    if (local_src) {
        auto sa = codec::VolumeArchive::open(root);
        if (!sa) return std::unexpected(sa.error());
        if (sa->src_dtype() != codec::DType::u8)
            return fenix::err(Errc::unsupported, "predict-scroll: only u8 .fxvol CT sources supported");
        shape = sa->dims();
    } else {
        auto meta = io::read_zarray(lroot);
        if (!meta) return std::unexpected(meta.error());
        shape = meta->shape;
        if (io::detail::dtype_size(meta->dtype) != 1)
            return fenix::err(Errc::unsupported,
                              "predict-scroll: only u8 (|u1) zarr sources supported, got " + meta->dtype);
    }

    // GPU model replicas — gpuworkers per visible device (falls back to a single CPU replica if no
    // CUDA/HIP). With >1 workers per device, one worker's gather/encode overlaps another's compute.
    const int ndev = torch::cuda::is_available() ? static_cast<int>(torch::cuda::device_count()) : 1;
    const int nworkers = ndev * static_cast<int>(torch::cuda::is_available() ? gpu_workers : 1);
    // Weights are .fxweights (eager ResEncUNet replicas), an AOTInductor .pt2 package
    // (torch.compile's fused kernels — aoti_net.hpp), or a TensorRT .plan/.engine
    // (trt_engine.hpp, FENIX_TRT builds only). One replica per worker; the worker below
    // is generic over the net type.
    const bool aoti_src = wpath.size() > 4 && wpath.ends_with(".pt2");
    const bool trt_src = wpath.size() > 4 && (wpath.ends_with(".plan") || wpath.ends_with(".engine"));
#ifndef FENIX_TRT
    if (trt_src) return fenix::err(Errc::unsupported, "predict-scroll: .plan/.engine needs a FENIX_TRT build");
#endif
    std::vector<nets::ResEncUNet> replicas;
    std::vector<AotiNet> aoti_replicas;
#ifdef FENIX_TRT
    std::vector<TrtNet> trt_replicas;
#endif
    std::vector<torch::Device> devs;
    for (int g = 0; g < nworkers; ++g) {
        torch::Device dev = torch::cuda::is_available()
                                ? torch::Device(torch::kCUDA, static_cast<torch::DeviceIndex>(g % ndev))
                                : torch::Device(torch::kCPU);
        if (aoti_src) {
            auto an = AotiNet::load(wpath, dev);
            if (!an) return std::unexpected(an.error());
            aoti_replicas.push_back(std::move(*an));
#ifdef FENIX_TRT
        } else if (trt_src) {
            // Each replica owns its engine/context, deserialized with ITS device current (TRT binds
            // the engine to the device at deserialize time, and enqueueV3 runs on the current
            // device's stream — unlike torch ops, which dispatch by tensor device).
            if (!dev.is_cuda()) return fenix::err(Errc::unsupported, "predict-scroll: trt engine needs CUDA");
            c10::cuda::set_device(dev.index());  // NOT CUDAGuard — its impl inlines raw cudart symbols
            auto tn = TrtNet::load(wpath);
            if (!tn) return std::unexpected(tn.error());
            trt_replicas.push_back(std::move(*tn));
#endif
        } else {
            // net->to(dev)/load_into throw on CUDA OOM (e.g. the device is occupied by another
            // process) — same exception boundary as run_predict's setup: catch, never terminate.
            try {
                nets::ResEncUNet net(nets::ResEncUNetConfig{});
                net->to(dev);
                net->eval();
                auto w = load_fxweights(wpath, dev);
                if (!w) return std::unexpected(w.error());
                std::vector<std::string> missing;
                load_into(*net, *w, &missing);
                if (!missing.empty())
                    return fenix::err(Errc::decode_error,
                                      "predict-scroll: weights/arch mismatch on device " + std::to_string(g));
                replicas.push_back(net);
            } catch (const std::exception& e) {
                return fenix::err(Errc::internal,
                                  "predict-scroll: model load on device " + std::to_string(g) + " failed: " + e.what());
            }
        }
        devs.push_back(dev);
    }
    if (aoti_src) {
        // The package's static geometry is authoritative (a mismatched patch would silently change
        // the tiling; a mismatched batch would pad every forward) — same rule as the TRT engine.
        if (patch != aoti_replicas[0].patch() || batch != static_cast<int>(aoti_replicas[0].batch()))
            fenix::log(LogLevel::info,
                       "predict-scroll: aoti package overrides patch {}->{} batch {}->{}",
                       patch,
                       aoti_replicas[0].patch(),
                       batch,
                       aoti_replicas[0].batch());
        patch = static_cast<int>(aoti_replicas[0].patch());
        batch = static_cast<int>(aoti_replicas[0].batch());
    }
#ifdef FENIX_TRT
    if (trt_src) {
        if (patch != static_cast<int>(trt_replicas[0].patch()) || batch != static_cast<int>(trt_replicas[0].batch()))
            fenix::log(LogLevel::info,
                       "predict-scroll: trt engine overrides patch {}->{} batch {}->{}",
                       patch,
                       trt_replicas[0].patch(),
                       batch,
                       trt_replicas[0].batch());
        patch = static_cast<int>(trt_replicas[0].patch());
        batch = static_cast<int>(trt_replicas[0].batch());
    }
#endif

    // Coarse occupancy map for air-skip (this is a masked volume — most of the box is air). Load the
    // coarsest available pyramid level; skip a region only if NO coarse voxel in its footprint is non-zero.
    Volume<u8> occ;
    s64 occ_scale = 1;
    if (local_src) {
        // Air-skip from the archive's LOD0 COVERAGE tri-state — one bit per 64³ chunk (Real/Zero present vs
        // Absent). This is the authoritative occupancy and needs no coarse LOD: export-scroll populates only
        // LOD0 (the coarse pyramid is unbuilt / all-absent), so reading a coarse level would read all-zero and
        // wrongly air-skip everything. Build a per-chunk occupancy volume at chunk granularity (occ_scale=T).
        auto sa = codec::VolumeArchive::open(root);
        if (!sa) return std::unexpected(sa.error());
        const ChunkCoord cs = sa->chunk_extent(0);  // number of 64³ chunks per axis
        occ = Volume<u8>(Extent3{cs.z, cs.y, cs.x});
        auto ov = occ.view();
        for (s64 cz = 0; cz < cs.z; ++cz)
            for (s64 cy = 0; cy < cs.y; ++cy)
                for (s64 cx = 0; cx < cs.x; ++cx)
                    ov(cz, cy, cx) = (sa->coverage(0, {cz, cy, cx}) == codec::Coverage::Real) ? u8{1} : u8{0};
        occ_scale = T;  // each occupancy voxel covers one 64³ chunk of source voxels
    } else {
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
    }
    auto occupied = [&](Index3 org, Extent3 ext) -> bool {
        if (occ.dims().z == 0) return true;
        const Extent3 os = occ.dims();
        auto ov = occ.view();
        const s64 z0 = std::max<s64>(0, org.z / occ_scale - 1),
                  z1 = std::min(os.z, (org.z + ext.z + occ_scale - 1) / occ_scale + 1);
        const s64 y0 = std::max<s64>(0, org.y / occ_scale - 1),
                  y1 = std::min(os.y, (org.y + ext.y + occ_scale - 1) / occ_scale + 1);
        const s64 x0 = std::max<s64>(0, org.x / occ_scale - 1),
                  x1 = std::min(os.x, (org.x + ext.x + occ_scale - 1) / occ_scale + 1);
        for (s64 z = z0; z < z1; ++z)
            for (s64 y = y0; y < y1; ++y)
                for (s64 x = x0; x < x1; ++x)
                    if (ov(z, y, x) != 0) return true;
        return false;
    };

    // ---- GLOBAL (zero-waste) mode ----
    // One patch grid over the whole bbox: every patch is computed exactly ONCE (no region halo, no
    // discarded compute — the region mode wastes 20-80% of its forwards on halo, surface/volume).
    // Patches Gaussian-scatter into a sparse chunk-keyed accumulator (ml/global_accum.hpp); a z-sweep
    // finalizes chunk rows behind the last patch row that can still touch them (normalize by the
    // factorized global weight profiles → u8 → archive → evict), so live RAM is one ~(P + ahead·S)
    // z-band of the occupied cross-section, never the volume. Resume: contiguous finalized chunk rows
    // (no Absent chunk across the bbox row) set the floor; the re-run starts at the first patch row
    // overlapping unfinalized voxels and scatters above the floor only. CT: local .fxvol → per-worker
    // read handles; zarr → ONE shared CachedVolume (each CT chunk fetched exactly once, and the cache
    // IS the ct= raw-CT archive — the region mode's separate ct writer becomes unnecessary here).
    if (mode == "global") {
        const int P = patch;
        const s64 S = std::max<s64>(1, static_cast<s64>(P * (1.0 - overlap)));
        // 64-align the bbox to the archive chunk grid (org down, end up), clamped to the volume.
        Index3 gorg{bb_org.z / T * T, bb_org.y / T * T, bb_org.x / T * T};
        const Index3 bbe = bb_ext.z > 0 ? Index3{bb_org.z + bb_ext.z, bb_org.y + bb_ext.y, bb_org.x + bb_ext.x}
                                        : Index3{shape.z, shape.y, shape.x};
        const Index3 gend{std::min(shape.z, (bbe.z + T - 1) / T * T),
                          std::min(shape.y, (bbe.y + T - 1) / T * T),
                          std::min(shape.x, (bbe.x + T - 1) / T * T)};
        const Extent3 gext{gend.z - gorg.z, gend.y - gorg.y, gend.x - gorg.x};
        if (gext.z <= 0 || gext.y <= 0 || gext.x <= 0)
            return fenix::err(Errc::invalid_argument, "predict-scroll: empty bbox");

        const bool resume = std::filesystem::exists(out_pred);
        auto pred_a = resume ? codec::VolumeArchive::open(out_pred, true)
                             : codec::VolumeArchive::create(out_pred, shape, codec::DctParams{.q = predq});
        if (!pred_a) return std::unexpected(pred_a.error());
        if (resume && !(pred_a->dims() == shape))
            return fenix::err(Errc::invalid_argument, "predict-scroll resume: pred archive dims != source shape");

        // Zarr CT goes through ONE shared CachedVolume (thread-safe fills; each chunk fetched once).
        std::optional<io::CachedVolume> ct_cv;
        if (!local_src) {
            const std::string cache = ct_path.empty() ? out_pred + ".ctcache.fxvol" : ct_path;
            auto cv = io::CachedVolume::open(cache, lroot, ctq);
            if (!cv) return std::unexpected(cv.error());
            ct_cv = std::move(*cv);
        } else if (!ct_path.empty()) {
            fenix::log(LogLevel::warn, "predict-scroll global: ct= ignored for a local .fxvol source");
        }

        // Global grid + factorized Gaussian weight profiles (bbox-local coords).
        const auto zs = tile_starts(gext.z, P, overlap);
        const auto ys = tile_starts(gext.y, P, overlap);
        const auto xs = tile_starts(gext.x, P, overlap);
        const auto g1 = gaussian1d(P);
        GlobalAccum accum(gorg,
                          gext,
                          global_weight_profile(g1, zs, gext.z, P),
                          global_weight_profile(g1, ys, gext.y, P),
                          global_weight_profile(g1, xs, gext.x, P));
        // Resume floor: chunk rows finalized bottom-up (a finalized row has NO Absent chunk in the bbox).
        const s64 ncz = accum.chunk_rows(), ncy = (gext.y + T - 1) / T, ncx = (gext.x + T - 1) / T;
        s64 floor_row = 0;
        if (resume) {
            for (; floor_row < ncz; ++floor_row) {
                bool full = true;
                for (s64 cy = 0; cy < ncy && full; ++cy)
                    for (s64 cx = 0; cx < ncx; ++cx)
                        if (pred_a->coverage(0, {gorg.z / T + floor_row, gorg.y / T + cy, gorg.x / T + cx}) ==
                            codec::Coverage::Absent) {
                            full = false;
                            break;
                        }
                if (!full) break;
            }
            accum.set_floor_row(floor_row);
        }
        const s64 floor_z = floor_row * T;
        // Patch list, row-ordered. Rows whose span is fully below the floor are already done; air
        // patches (no Real CT chunk in the footprint) are dropped — the profiles still count them,
        // so their (air) voxels normalize low, matching the region mode's tile_filter semantics.
        struct GPatch {
            s32 row;
            s64 z0, y0, x0;  // bbox-local
        };
        std::vector<GPatch> patches;
        std::vector<s64> row_left(zs.size(), 0);
        s64 first_row = static_cast<s64>(zs.size());
        for (usize i = 0; i < zs.size(); ++i) {
            if (zs[i] + P <= floor_z) continue;  // fully finalized — nothing above the floor to redo
            first_row = std::min(first_row, static_cast<s64>(i));
            for (s64 y0 : ys)
                for (s64 x0 : xs)
                    if (occupied({gorg.z + zs[i], gorg.y + y0, gorg.x + x0}, {P, P, P})) {
                        patches.push_back({static_cast<s32>(i), zs[i], y0, x0});
                        ++row_left[i];
                    }
        }
        const s64 nrows = static_cast<s64>(zs.size());
        const s64 npatch = static_cast<s64>(patches.size());
        const int ne = tta <= 1 ? 1 : (tta < 48 ? tta : 48);
        const int G = ne > 1 ? std::max(1, std::min(batch / ne, 16)) : std::max(1, batch);
        // Eager replicas run fp16 forwards (the region path converts inside predict_surface_filled,
        // which global mode bypasses); AotiNet/TrtNet to() are no-ops.
        for (auto& r : replicas) r->to(torch::kFloat16);
        fenix::log(LogLevel::info,
                   "predict-scroll GLOBAL {} -> {}  bbox [{},{},{}]+[{},{},{}]  {} patches ({} rows, stride {}, "
                   "{} GPU worker(s), group {}) tta={} batch={} patch={} floor-row {}/{}",
                   root,
                   out_pred,
                   gorg.z,
                   gorg.y,
                   gorg.x,
                   gext.z,
                   gext.y,
                   gext.x,
                   npatch,
                   nrows,
                   S,
                   nworkers,
                   G,
                   tta,
                   batch,
                   P,
                   floor_row,
                   ncz);

        // Shared sweep state. Workers claim row-ordered patch groups but never run more than
        // ahead_rows past the finalize front (bounds the live accumulator band).
        std::mutex mtx;
        std::condition_variable cv_claim, cv_front;
        s64 cursor = 0, front_row = first_row;
        std::atomic<s64> patches_done{0};
        bool cancel = false, has_err = false;
        int workers_live = nworkers;
        Error worker_err;

        auto gworker = [&](int g, auto& net) {
            torch::NoGradGuard ng;
#ifdef FENIX_TRT
            // TRT enqueues on the CURRENT device's stream (torch ops dispatch by tensor device and
            // don't need this) — pin this worker thread to its replica's device. set_device, not
            // CUDAGuard: the guard's impl inlines raw cudart symbols fenix doesn't link.
            if (trt_src && devs[static_cast<usize>(g)].is_cuda())
                c10::cuda::set_device(devs[static_cast<usize>(g)].index());
#endif
            std::optional<codec::VolumeArchive> src_arch;
            if (local_src) {
                auto sa = codec::VolumeArchive::open(root);
                if (!sa) {
                    std::unique_lock lk(mtx);
                    if (!has_err) {
                        worker_err = sa.error();
                        has_err = true;
                    }
                    cancel = true;
                    cv_front.notify_all();
                    cv_claim.notify_all();
                    --workers_live;
                    return;
                }
                src_arch = std::move(*sa);
            }
            const std::size_t PN = static_cast<std::size_t>(P) * P * P;
            auto fail = [&](Error e) {
                std::unique_lock lk(mtx);
                if (!has_err) {
                    worker_err = std::move(e);
                    has_err = true;
                }
                cancel = true;
                cv_front.notify_all();
                cv_claim.notify_all();
            };
            // Claim the next row-ordered patch group (bounded by the finalize front + ahead_rows).
            // Returns false when the work list is exhausted or the run is canceled.
            auto claim = [&](std::vector<GPatch>& out) {
                out.clear();
                std::unique_lock lk(mtx);
                cv_claim.wait(lk, [&] {
                    return cancel || cursor >= npatch ||
                           patches[static_cast<usize>(cursor)].row <= front_row + ahead_rows;
                });
                if (cancel || cursor >= npatch) return false;
                while (static_cast<int>(out.size()) < G && cursor < npatch &&
                       patches[static_cast<usize>(cursor)].row <= front_row + ahead_rows)
                    out.push_back(patches[static_cast<usize>(cursor++)]);
                return true;
            };
            // Gather + z-score one group into fb; lv = indices that survived the constant-CT skip.
            // Runs on the prefetch thread while the previous group is on the GPU.
            auto gather = [&](const std::vector<GPatch>& grp,
                              std::vector<float>& fb,
                              std::vector<usize>& lv,
                              std::vector<u8>& tmpb) {
                lv.clear();
                for (usize k = 0; k < grp.size(); ++k) {
                    const GPatch& p = grp[k];
                    const s64 z0 = gorg.z + p.z0, y0 = gorg.y + p.y0, x0 = gorg.x + p.x0;
                    // Clip the read to the volume, replicate edges into the P³ f32 patch (same
                    // edge-clamp semantics as the region path's at_clamped filler).
                    const s64 D = std::min<s64>(P, shape.z - z0), H = std::min<s64>(P, shape.y - y0),
                              W = std::min<s64>(P, shape.x - x0);
                    Expected<void> gr = fenix::err(Errc::internal, "unset");
                    if (local_src)
                        gr = src_arch->gather_box_u8(0, z0, y0, x0, D, H, W, tmpb.data());
                    else
                        gr = ct_cv->gather_box_u8(z0, y0, x0, D, H, W, tmpb.data());
                    if (!gr) {
                        fail(gr.error());
                        return false;
                    }
                    // Constant CT (masked/air fill) → z-score would divide by ~0 and the net emits
                    // structured garbage; skip (contributes nothing — normalizes as air).
                    const u8 first = tmpb[0];
                    bool constant = true;
                    for (s64 i = 0, n = D * H * W; i < n; ++i)
                        if (tmpb[static_cast<usize>(i)] != first) {
                            constant = false;
                            break;
                        }
                    if (constant) continue;
                    float* dst = fb.data() + PN * lv.size();
                    parallel_for(0, P, [&](s64 z) {
                        const s64 sz = std::min(z, D - 1);
                        for (s64 y = 0; y < P; ++y) {
                            const s64 sy = std::min(y, H - 1);
                            const u8* srow = tmpb.data() + static_cast<usize>((sz * H + sy) * W);
                            float* drow = dst + (static_cast<usize>(z) * P + static_cast<usize>(y)) * P;
                            for (s64 x = 0; x < P; ++x) drow[x] = static_cast<float>(srow[std::min(x, W - 1)]);
                        }
                    });
                    detail::norm_patch(dst, P, Norm::zscore);
                    lv.push_back(k);
                }
                return true;
            };
            // Double-buffered pipeline: a prefetch thread claims + gathers group N+1 (CPU/S3) while
            // group N runs on the GPU — the gather latency (dominant on zarr sources, material on
            // local decode) hides behind the forward. Row accounting for group N happens BEFORE
            // joining the prefetch (its claim may need the finalize front to advance).
            std::vector<u8> tmpA(PN), tmpB(PN);
            std::vector<float> fbufA(PN * static_cast<usize>(G)), fbufB(PN * static_cast<usize>(G));
            std::vector<GPatch> cur, nxt;
            std::vector<usize> liveA, liveB;
            bool more = claim(cur);
            bool okA = more && gather(cur, fbufA, liveA, tmpA);
            while (more) {
                bool have_nxt = false, okB = false;
                std::thread pf([&] {
                    have_nxt = claim(nxt);
                    if (have_nxt) okB = gather(nxt, fbufB, liveB, tmpB);
                });
                bool fwd_ok = okA;
                if (okA && !liveA.empty()) {
                    try {
                        const int nk = static_cast<int>(liveA.size());
                        auto xb = torch::from_blob(fbufA.data(), {nk, 1, P, P, P}, torch::kFloat32);
                        auto xin = xb.to(torch::kFloat16).to(devs[static_cast<usize>(g)]);
                        auto surf = detail::d2h_prob(detail::tta_infer_dev(
                            net, xin, ne, batch, /*sigmoid=*/false, /*channel=*/1));  // [nk,P,P,P] f32
                        const float* base = surf.template data_ptr<float>();
                        for (int k = 0; k < nk; ++k) {
                            const GPatch& p = cur[liveA[static_cast<usize>(k)]];
                            accum.scatter(gorg.z + p.z0,
                                          gorg.y + p.y0,
                                          gorg.x + p.x0,
                                          P,
                                          base + PN * static_cast<usize>(k),
                                          g1,
                                          g1,
                                          g1);
                        }
                    } catch (const std::exception& e) {
                        fail(fenix::err(Errc::internal,
                                        std::string("predict-scroll global: forward failed: ") + e.what())
                                 .error());
                        fwd_ok = false;
                    }
                }
                {
                    std::unique_lock lk(mtx);
                    for (const GPatch& p : cur) --row_left[static_cast<usize>(p.row)];
                    cv_front.notify_all();
                }
                patches_done += static_cast<s64>(cur.size());
                pf.join();
                if (!fwd_ok) break;
                more = have_nxt;
                cur.swap(nxt);
                liveA.swap(liveB);
                fbufA.swap(fbufB);
                std::swap(tmpA, tmpB);
                okA = okB;
            }
            std::unique_lock lk(mtx);
            --workers_live;
            cv_front.notify_all();
        };

        std::vector<std::thread> pool;
        for (int g = 0; g < nworkers; ++g)
            pool.emplace_back([&, g] {
                if (aoti_src)
                    gworker(g, aoti_replicas[static_cast<usize>(g)]);
#ifdef FENIX_TRT
                else if (trt_src)
                    gworker(g, trt_replicas[static_cast<usize>(g)]);
#endif
                else
                    gworker(g, replicas[static_cast<usize>(g)]);
            });

        // Finalize/writer loop (single thread — the archive's one writer). Advances the front as rows
        // complete, finalizes chunk rows strictly below the next pending patch row, commits, logs.
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        auto t_last = t0;
        u64 real_tiles = 0, zero_tiles = 0;
        Expected<void> write_err;
        auto sink = [&](ChunkCoord tc, const u8* blk) {
            if (!write_err) return;
            auto w = pred_a->write_chunk(0, tc, std::span<const u8>(blk, static_cast<usize>(T * T * T)));
            if (!w) {
                write_err = std::unexpected(w.error());
                return;
            }
            (pred_a->coverage(0, tc) == codec::Coverage::Real ? real_tiles : zero_tiles)++;
        };
        for (;;) {
            {
                std::unique_lock lk(mtx);
                cv_front.wait_for(lk, std::chrono::seconds(15), [&] {
                    return cancel || workers_live == 0 ||
                           (front_row < nrows && row_left[static_cast<usize>(front_row)] == 0);
                });
                if (cancel) break;
                while (front_row < nrows && row_left[static_cast<usize>(front_row)] == 0) ++front_row;
                cv_claim.notify_all();
            }
            const s64 row_lim = front_row >= nrows ? ncz : zs[static_cast<usize>(front_row)] / T;
            if (row_lim > accum.floor_row()) {
                accum.finalize_rows_below(row_lim, sink);
                if (!write_err) break;
                if (auto c = pred_a->commit(); !c) {
                    write_err = std::unexpected(c.error());
                    break;
                }
            }
            if (const auto nowt = clk::now(); std::chrono::duration<f64>(nowt - t_last).count() >= 15.0) {
                t_last = nowt;
                const f64 el = std::chrono::duration<f64>(nowt - t0).count();
                const s64 done = patches_done.load();
                const f64 frac = static_cast<f64>(done) / static_cast<f64>(npatch ? npatch : 1);
                fenix::log(LogLevel::info,
                           "  global {}/{} patches ({:.1f}%) row {}/{} {:.0f}s ETA {:.0f}s | real {} zero {} | "
                           "band {} blocks ({:.1f} GiB) | {:.1f} MiB out",
                           done,
                           npatch,
                           100.0 * frac,
                           front_row,
                           nrows,
                           el,
                           frac > 0 ? el * (1.0 - frac) / frac : 0.0,
                           real_tiles,
                           zero_tiles,
                           accum.live_blocks(),
                           static_cast<f64>(accum.live_blocks()) * T * T * T * 4 / (1024.0 * 1024.0 * 1024.0),
                           static_cast<f64>(pred_a->committed_size()) / (1024.0 * 1024.0));
            }
            std::unique_lock lk(mtx);
            if (front_row >= nrows || (workers_live == 0 && cursor >= npatch)) break;
        }
        {
            std::unique_lock lk(mtx);
            if (has_err || !write_err) cancel = true;  // error → wake + stop workers; success → they drain
            cv_claim.notify_all();
        }
        for (auto& th : pool) th.join();
        if (has_err) return std::unexpected(worker_err);
        if (!write_err) return std::unexpected(write_err.error());
        // Tail: everything is complete — finalize any remaining rows and seal.
        accum.finalize_rows_below(ncz, sink);
        if (!write_err) return std::unexpected(write_err.error());
        if (auto c = pred_a->commit(); !c) return std::unexpected(c.error());
        if (auto c = pred_a->close(); !c) return std::unexpected(c.error());
        fenix::log(LogLevel::info,
                   "predict-scroll GLOBAL done: {} patches, real {} / zero {} tiles, {:.1f} MiB pred, {:.0f}s",
                   npatch,
                   real_tiles,
                   zero_tiles,
                   static_cast<f64>(pred_a->committed_size()) / (1024.0 * 1024.0),
                   std::chrono::duration<f64>(clk::now() - t0).count());
        return 0;
    }

    // Create/open the output archives at FULL scroll dims (sparse page table + coverage sentinels).
    // Resume: open an EXISTING archive writable (COW preserves the committed snapshot); create() O_TRUNCs.
    const bool resume = std::filesystem::exists(out_pred);
    auto pred_a = resume ? codec::VolumeArchive::open(out_pred, true)
                         : codec::VolumeArchive::create(out_pred, shape, codec::DctParams{.q = predq});
    if (!pred_a) return std::unexpected(pred_a.error());
    if (resume && !(pred_a->dims() == shape))
        return fenix::err(Errc::invalid_argument, "predict-scroll resume: pred archive dims != source shape");
    std::optional<codec::VolumeArchive> ct_a;
    if (!ct_path.empty()) {
        const bool ct_resume = std::filesystem::exists(ct_path);
        auto c = ct_resume ? codec::VolumeArchive::open(ct_path, true)
                           : codec::VolumeArchive::create(ct_path, shape, codec::DctParams{.q = ctq});
        if (!c) return std::unexpected(c.error());
        if (ct_resume && !(c->dims() == shape))
            return fenix::err(Errc::invalid_argument, "predict-scroll resume: ct archive dims != source shape");
        ct_a = std::move(*c);
    }

    // Region grid — limited to the requested sub-box (default = whole volume). Snap the box to the region
    // grid so tiles align to R (and chunk) boundaries.
    const s64 lz0 = std::clamp<s64>(bb_org.z, 0, shape.z), ly0 = std::clamp<s64>(bb_org.y, 0, shape.y),
              lx0 = std::clamp<s64>(bb_org.x, 0, shape.x);
    const s64 lz1 = bb_ext.z < 0 ? shape.z : std::min(shape.z, lz0 + bb_ext.z);
    const s64 ly1 = bb_ext.y < 0 ? shape.y : std::min(shape.y, ly0 + bb_ext.y);
    const s64 lx1 = bb_ext.x < 0 ? shape.x : std::min(shape.x, lx0 + bb_ext.x);
    const s64 rz_lo = lz0 / R, rz_hi = ndiv(lz1, R), ry_lo = ly0 / R, ry_hi = ndiv(ly1, R), rx_lo = lx0 / R,
              rx_hi = ndiv(lx1, R);
    const s64 total = (rz_hi - rz_lo) * (ry_hi - ry_lo) * (rx_hi - rx_lo);

    // Worklist: occupied, not-yet-done regions (resume-skip on the PREDICTION archive's coverage).
    struct Work {
        Index3 org;
        Extent3 ext;
    };
    std::vector<Work> work;
    s64 skipped = 0, skipped_air = 0;
    for (s64 rz = rz_lo; rz < rz_hi; ++rz)
        for (s64 ry = ry_lo; ry < ry_hi; ++ry)
            for (s64 rx = rx_lo; rx < rx_hi; ++rx) {
                const Index3 org{rz * R, ry * R, rx * R};
                const Extent3 ext{
                    std::min(R, shape.z - org.z), std::min(R, shape.y - org.y), std::min(R, shape.x - org.x)};
                const ChunkCoord ft{org.z / T, org.y / T, org.x / T};
                if (pred_a->coverage(0, ft) != codec::Coverage::Absent) {
                    ++skipped;
                    continue;
                }
                if (!occupied(org, ext)) {
                    ++skipped_air;
                    continue;
                }
                work.push_back({org, ext});
            }
    const s64 nwork = static_cast<s64>(work.size());
    fenix::log(LogLevel::info,
               "predict-scroll {} L{} ({}x{}x{}) -> {}{}  {} GPU(s) x{} worker(s) tta={} batch={} patch={} "
               "region={}  {} regions to run ({} air-skip, {} resume-skip of {})  {}",
               root,
               level,
               shape.z,
               shape.y,
               shape.x,
               out_pred,
               ct_path.empty() ? "" : (" + " + ct_path).c_str(),
               ndev,
               gpu_workers,
               tta,
               batch,
               patch,
               R,
               nwork,
               skipped_air,
               skipped,
               total,
               resume ? "RESUME" : "fresh");

    // Halo so a region's border patches see context from the neighbours (avoids seam artifacts): the model
    // sees [org-halo, org+ext+halo], predicts, and only the core [org, org+ext] is written. Default halo =
    // patch/2 = one patch-radius each side, which is the MINIMAL width for a seamless overlap-blend: with
    // stride patch*(1-overlap), a core-boundary voxel is only covered by the same full set of Gaussian-
    // weighted patches as an interior voxel once patches can center up to patch/2 beyond the core edge.
    // A smaller halo trades a faint region-seam for less discarded compute; `halo=` overrides. Rounded to
    // the 64³ tile grid. NOTE: the halo is a fixed shell, so its COST fraction shrinks as region grows —
    // region=512 gathers 768³ (125 patches, ~78% halo) but region=2048 gathers 2304³ (~42% overhead). The
    // big lever is region size, not shrinking the halo.
    const s64 halo = halo_arg >= 0 ? ((halo_arg + T - 1) / T) * T : ((patch / 2 + T - 1) / T) * T;
    {
        const s64 gext = R + 2 * halo;  // gather-box side for an interior region
        const f64 waste = 1.0 - (static_cast<f64>(R) * R * R) / (static_cast<f64>(gext) * gext * gext);
        fenix::log(LogLevel::info,
                   "  halo={} → interior gather {}³ per {}³ core ({:.0f}% discarded-halo compute)",
                   halo,
                   gext,
                   R,
                   100.0 * waste);
    }

    // Producer/consumer with a GPU-compute middle. Each GPU worker owns device g and pulls regions off the
    // shared cursor; it gathers CT (with halo, clamped to the volume), runs TTA inference on device g, and
    // pushes the finished (core CT block, core pred block) to the writer queue. The single writer thread
    // (this thread) drains the queue into the archives.
    struct Result {
        s64 idx;
        Volume<u8> ct_core;
        Volume<u8> pred_core;
    };
    std::mutex mtx;
    std::condition_variable cv_push, cv_pop;
    std::deque<Result> queue;
    const usize qcap = static_cast<usize>(ndev) + 1;
    std::atomic<s64> cursor{0};
    int workers_live = nworkers;
    bool cancel = false;
    Error worker_err;
    bool has_err = false;

    InferOptions iopt;
    iopt.patch = patch;
    iopt.overlap = overlap;
    iopt.tta = tta;
    iopt.batch = std::max(1, batch);

    auto worker = [&](int g, auto& net) {
        // No explicit CUDAGuard: each replica already lives on devs[g] and predict_surface_filled moves
        // every patch to that device, so libtorch dispatches all ops on the correct device per worker.
        // (An OptionalCUDAGuard would pull raw cudart symbols the rest of fenix deliberately doesn't link.)
        torch::NoGradGuard ng;
        // Local .fxvol source: each worker opens its OWN read-only archive handle on the same file. Multiple
        // read-only handles on one file are safe; this avoids any cross-thread sharing of a single archive's
        // mutable block cache (belt-and-suspenders over the archive's documented-thread-safe reads).
        std::optional<codec::VolumeArchive> src_arch;
        if (local_src) {
            auto sa = codec::VolumeArchive::open(root);
            if (!sa) {
                std::unique_lock lk(mtx);
                if (!has_err) {
                    worker_err = sa.error();
                    has_err = true;
                }
                cancel = true;
                cv_pop.notify_all();
                return;
            }
            src_arch = std::move(*sa);
        }
        for (;;) {
            const s64 i = cursor.fetch_add(1);
            if (i >= nwork) break;
            {
                std::unique_lock lk(mtx);
                if (cancel) break;
            }
            const Work w = work[static_cast<usize>(i)];
            // Halo-expanded gather box, clamped to the volume.
            const Index3 gorg{
                std::max<s64>(0, w.org.z - halo), std::max<s64>(0, w.org.y - halo), std::max<s64>(0, w.org.x - halo)};
            const Index3 gend{std::min(shape.z, w.org.z + w.ext.z + halo),
                              std::min(shape.y, w.org.y + w.ext.y + halo),
                              std::min(shape.x, w.org.x + w.ext.x + halo)};
            const Extent3 gext{gend.z - gorg.z, gend.y - gorg.y, gend.x - gorg.x};
            // Gather CT. Local .fxvol → decode the halo box from disk (hard-error on failure; no retry loop —
            // a local decode error isn't transient). Zarr → fetch, riding through transient S3 blips with
            // capped backoff (a multi-day run must not die on one GET). The gather box is already clamped to
            // the volume, so gather_box_u8's in-bounds precondition holds.
            Expected<Volume<u8>> ctv = fenix::err(Errc::internal, "unset");
            if (local_src) {
                Volume<u8> cv(gext);
                auto r = src_arch->gather_box_u8(0, gorg.z, gorg.y, gorg.x, gext.z, gext.y, gext.x, cv.data());
                if (!r)
                    ctv = std::unexpected(r.error());
                else
                    ctv = std::move(cv);
            } else {
                ctv = io::read_zarr_region<u8>(lroot, gorg, gext);
                for (int attempt = 1; !ctv && attempt <= 20; ++attempt) {
                    {
                        std::unique_lock lk(mtx);
                        if (cancel) break;
                    }
                    const int backoff = std::min(30, 1 << std::min(attempt, 5));
                    fenix::log(LogLevel::warn,
                               "predict-scroll region z{} y{} x{} CT read failed: {} — retry {}/20 in {}s",
                               gorg.z,
                               gorg.y,
                               gorg.x,
                               ctv.error().message,
                               attempt,
                               backoff);
                    std::this_thread::sleep_for(std::chrono::seconds(backoff));
                    ctv = io::read_zarr_region<u8>(lroot, gorg, gext);
                }
            }
            if (!ctv) {
                std::unique_lock lk(mtx);
                if (!has_err) {
                    worker_err = ctv.error();
                    has_err = true;
                }
                cancel = true;
                cv_pop.notify_all();
                break;
            }
            // Octahedral-TTA sliding-window inference on THIS device, gathering each patch straight from the
            // dense u8 CT (widened to f32 per patch, edge-clamped) — the same filler predict_surface_window
            // uses. The output CROP (core_org/core_ext) makes predict_surface_filled size its accumulator to
            // the CORE (not the halo-expanded gather box) and skip pure-halo patches: RAM ∝ core³ not gather³
            // (removes the OOM ceiling at large region=), and no wasted halo compute. Returns a core-sized vol.
            const Index3 coff{w.org.z - gorg.z, w.org.y - gorg.y, w.org.x - gorg.x};  // core offset in the gather box
            auto ctview = ctv->view();
            InferOptions wopt = iopt;  // per-region copy — never mutate the shared iopt across workers
            wopt.core_org = coff;
            wopt.core_ext = w.ext;
            // Masked/air tiles are CONSTANT CT (masked chunks decode to a uniform fill) — z-score
            // then divides by ~0 and the net emits structured garbage (seen as a faint prediction
            // grid over masked background). Skip any all-constant tile: unwritten prob stays 0.
            wopt.tile_filter = [ctview](s64 z0, s64 y0, s64 x0, s64 P) {
                const Extent3 d = ctview.dims();
                const s64 z1 = std::min(d.z, z0 + P), y1 = std::min(d.y, y0 + P), x1 = std::min(d.x, x0 + P);
                const u8 first = ctview(std::min(z0, d.z - 1), std::min(y0, d.y - 1), std::min(x0, d.x - 1));
                for (s64 z = z0; z < z1; ++z)
                    for (s64 y = y0; y < y1; ++y) {
                        const u8* row = &ctview(z, y, x0);
                        for (s64 x = 0; x < x1 - x0; ++x)
                            if (row[x] != first) return true;
                    }
                return false;  // constant input -> skip tile
            };
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
                net,
                devs[static_cast<usize>(g)],
                wopt);
            if (!probf) {
                std::unique_lock lk(mtx);
                if (!has_err) {
                    worker_err = probf.error();
                    has_err = true;
                }
                cancel = true;
                cv_pop.notify_all();
                break;
            }
            // probf is already the CORE [w.org, w.org+ext] (cropped by predict_surface_filled). Convert to u8;
            // CT core is cropped from the gather box at the same offset.
            Volume<u8> pred_core(w.ext), ct_core(w.ext);
            auto pf = probf->view();
            auto cin = ctv->view();
            auto pc = pred_core.view();
            auto cc = ct_core.view();
            for (s64 z = 0; z < w.ext.z; ++z)
                for (s64 y = 0; y < w.ext.y; ++y)
                    for (s64 x = 0; x < w.ext.x; ++x) {
                        pc(z, y, x) =
                            static_cast<u8>(std::clamp(pf(z, y, x), 0.0f, 1.0f) * 255.0f + 0.5f);  // probf IS the core
                        cc(z, y, x) = cin(z + coff.z, y + coff.y, x + coff.x);  // CT cropped from the gather box
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
    for (int g = 0; g < nworkers; ++g)
        pool.emplace_back([&, g] {
#ifdef FENIX_TRT
            // TRT enqueues on the CURRENT device's stream — pin the thread to its replica's device
            // (set_device, not CUDAGuard: the guard's impl inlines raw cudart symbols).
            if (trt_src && devs[static_cast<usize>(g)].is_cuda())
                c10::cuda::set_device(devs[static_cast<usize>(g)].index());
#endif
            if (aoti_src)
                worker(g, aoti_replicas[static_cast<usize>(g)]);
#ifdef FENIX_TRT
            else if (trt_src)
                worker(g, trt_replicas[static_cast<usize>(g)]);
#endif
            else
                worker(g, replicas[static_cast<usize>(g)]);
        });

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
                                blk[static_cast<usize>((z * T + y) * T + x)] = cv(std::min(tz * T + z, ext.z - 1),
                                                                                  std::min(ty * T + y, ext.y - 1),
                                                                                  std::min(tx * T + x, ext.x - 1));
                    const ChunkCoord tc{org.z / T + tz, org.y / T + ty, org.x / T + tx};
                    if (auto w = a.write_chunk(0, tc, std::span<const u8>(blk)); !w) {
                        write_err = std::unexpected(w.error());
                        return false;
                    }
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
            fenix::log(
                LogLevel::info,
                "  predict-scroll {}/{} regions ({:.1f}%) {:.0f}s ETA {:.0f}s | real {} zero {} tiles | {:.1f} MiB",
                done,
                nwork,
                100.0 * frac,
                el,
                frac > 0 ? el * (1.0 - frac) / frac : 0.0,
                real_tiles,
                zero_tiles,
                static_cast<f64>(pred_a->committed_size()) / (1024.0 * 1024.0));
        }
        const Index3 org = work[static_cast<usize>(r.idx)].org;
        const Extent3 ext = work[static_cast<usize>(r.idx)].ext;
        bool ok = write_core(*pred_a, r.pred_core, org, ext);
        if (ok)
            for (s64 tz = 0; tz < ndiv(ext.z, T); ++tz)
                for (s64 ty = 0; ty < ndiv(ext.y, T); ++ty)
                    for (s64 tx = 0; tx < ndiv(ext.x, T); ++tx)
                        (pred_a->coverage(0, {org.z / T + tz, org.y / T + ty, org.x / T + tx}) == codec::Coverage::Real
                             ? real_tiles
                             : zero_tiles)++;
        if (ok && ct_a) ok = write_core(*ct_a, r.ct_core, org, ext);
        if (ok && ++since_commit >= commit_every) {
            if (auto c = pred_a->commit(); !c) {
                write_err = std::unexpected(c.error());
                ok = false;
            }
            if (ok && ct_a) {
                if (auto c = ct_a->commit(); !c) {
                    write_err = std::unexpected(c.error());
                    ok = false;
                }
            }
            since_commit = 0;
        }
        ++done;
        if (!ok) {
            std::unique_lock lk(mtx);
            cancel = true;
            cv_push.notify_all();
            break;
        }
    }
    for (auto& th : pool) th.join();
    if (has_err) return std::unexpected(worker_err);
    if (!write_err) return std::unexpected(write_err.error());
    if (auto c = pred_a->close(); !c) return std::unexpected(c.error());
    if (ct_a) {
        if (auto c = ct_a->close(); !c) return std::unexpected(c.error());
    }
    fenix::log(
        LogLevel::info,
        "predict-scroll done: {} regions run ({} air-skip, {} resume-skip), real {} / zero {} tiles, {:.1f} MiB pred",
        nwork,
        skipped_air,
        skipped,
        real_tiles,
        zero_tiles,
        static_cast<f64>(pred_a->committed_size()) / (1024.0 * 1024.0));
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
