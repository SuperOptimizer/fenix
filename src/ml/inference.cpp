// ml/inference.cpp — THE libtorch translation unit: the ONLY place <torch/torch.h> is parsed. Compiled
// only for FENIX_ML builds; it defines the torch-free entry points declared in ml/ml_api.hpp. Isolating
// libtorch here (instead of inline in ml.hpp, which the driver pulled in) keeps the driver and every other
// TU torch-free — the unity driver TU drops from ~7 min to seconds. See ADR 0008 (build firewall).
#include "ml/ml_api.hpp"

#ifdef FENIX_ML

#include "codec/archive.hpp"
#include "io/nrrd.hpp"
#include "io/surface.hpp"
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
#include <chrono>
#include <functional>
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

// `fenix ml ink2d-raw <weights> <in.raw> <out.raw> <D> <H> <W>` — run the ResNet-152-3D ink
// model on a raw f32 segment subvolume, write the 2D ink logit. Validation hook vs reference.
static Expected<int> run_ink2d_raw(std::span<const std::string_view> a) {
    if (a.size() < 7)
        return fenix::err(Errc::invalid_argument, "usage: ml ink2d-raw <weights> <in.raw> <out.raw> <D> <H> <W>");
    const std::string wpath(a[1]), inpath(a[2]), outpath(a[3]);
    const long D = std::stol(std::string(a[4])), H = std::stol(std::string(a[5])), W = std::stol(std::string(a[6]));
    nets::ResNet3DInk net(true);
    const auto dev = best_device();
    net->to(dev);
    net->eval();
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
                                      Volume<f32>& nrrd_vol,
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
            srcv = nrrd_vol.view();
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
        prob = predict_surface(nrrd_vol.view(), net, dev, opt);
    }
    if (!prob) return std::unexpected(prob.error());

    // The dense source is done — release it before the output encode allocates its buffers
    // (1 GiB at 1024³, 8 GiB at 2048³ off the peak during the write phase).
    vol_u8 = Volume<u8>();
    nrrd_vol = Volume<f32>();
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
                              " <in.fxvol|.nrrd> <weights.fxweights> "
                              "<out.fxvol|.nrrd> [patch] [overlap] [tta] [batch]");
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
        return run_predict_core(name, inpath, wpath, outpath, opt, *tnet, dev0, d, vol_u8, nrrd_vol, u8_src);
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
            return run_predict_core(name, inpath, wpath, outpath, opt, jnet, dev0, d, vol_u8, nrrd_vol, u8_src);
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

    return run_predict_core(name, inpath, wpath, outpath, opt, net, dev, d, vol_u8, nrrd_vol, u8_src);
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
    // r152 (ink_canonical_2um): 256-tile, 62-layer window. r50: 64-tile stride-16 (75%
    // overlap), 30-layer window, small tiles -> big batch (upstream inference.py CFG).
    long tile = r50 ? 64 : 256, dwin = r50 ? 30 : 62, batch = r50 ? 128 : 2;
    double overlap = r50 ? 0.75 : 0.25;
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
