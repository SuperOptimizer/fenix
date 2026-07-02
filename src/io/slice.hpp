// io/slice.hpp — quick-look slice + video export. Pull a 2D slice from any axis of a volume,
// window it to 8-bit, optionally tint it with a RED overlay of a second volume (e.g. surface
// probability over raw CT), and save as JPEG (our encoder) or stream as frames to ffmpeg for a
// H.264 .mp4 (NVENC on the GPU if available, else libx264). ffmpeg is an external runtime tool
// invoked for the video mux/encode only — fenix links no video library.
#pragma once

#include "core/core.hpp"
#include "codec/archive.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"

#include <algorithm>
#include <charconv>
#include <csignal>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

namespace fenix::io {

// std::sto*/stof throw on bad input; the project builds with -fno-exceptions, so a raw sto* on
// untrusted CLI text aborts the process instead of returning a usage Error. Parse via from_chars
// (module convention, matches io.hpp's parse_i) and require the WHOLE token to be consumed —
// trailing garbage ("12abc") is rejected rather than silently truncated.
template <class T>
inline Expected<T> parse_num(std::string_view s) {
    T v{};
    const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
    if (ec != std::errc{} || ptr != s.data() + s.size())
        return err(Errc::invalid_argument, "not a number: \"" + std::string(s) + "\"");
    return v;
}

enum class Axis { z = 0, y = 1, x = 2 };

inline std::optional<Axis> parse_axis(std::string_view s) {
    if (s == "z" || s == "0") return Axis::z;
    if (s == "y" || s == "1") return Axis::y;
    if (s == "x" || s == "2") return Axis::x;
    return std::nullopt;
}

// Slice geometry: number of frames along the axis, and the (H,W) of each slice image.
struct SliceGeom { s64 frames, H, W; };
inline SliceGeom slice_geom(Extent3 d, Axis a) {
    switch (a) {
        case Axis::z: return {d.z, d.y, d.x};
        case Axis::y: return {d.y, d.z, d.x};
        default:      return {d.x, d.z, d.y};
    }
}
inline f32 slice_at(VolumeView<const f32> v, Axis a, s64 idx, s64 r, s64 c) {
    switch (a) {
        case Axis::z: return v(idx, r, c);
        case Axis::y: return v(r, idx, c);
        default:      return v(r, c, idx);
    }
}

struct SliceOpts {
    f32 vmin = 0, vmax = 0;        // window; if equal, auto from volume min/max
    f32 alpha = 0.6f;              // overlay strength
    f32 overlay_thresh = 0.0f;     // ignore overlay below this prob
    std::array<u8, 3> tint = {255, 0, 0};  // overlay color (default red)
};

inline std::array<u8, 3> parse_color(std::string_view s) {
    if (s == "red") return {255, 0, 0};
    if (s == "green") return {0, 255, 0};
    if (s == "blue") return {0, 96, 255};
    if (s == "cyan") return {0, 255, 255};
    if (s == "yellow") return {255, 255, 0};
    if (s == "magenta") return {255, 0, 255};
    return {255, 0, 0};
}

// Build one slice as an Image. If `prob` is set, output is RGB with a red overlay where prob is
// high; otherwise grayscale.  g = window(raw); R = g+(255-g)*a, G=B=g*(1-a), a = alpha*prob.
inline Image build_slice(VolumeView<const f32> raw, Axis a, s64 idx, const SliceOpts& o,
                         const VolumeView<const f32>* prob = nullptr) {
    const SliceGeom g = slice_geom(raw.dims(), a);
    const s64 H = g.H & ~s64{1}, W = g.W & ~s64{1};  // even dims (h264-friendly)
    const f32 lo = o.vmin, hi = (o.vmax > o.vmin) ? o.vmax : o.vmin + 1.0f;
    const f32 inv = 1.0f / (hi - lo);
    Image im;
    im.w = static_cast<int>(W);
    im.h = static_cast<int>(H);
    im.comps = prob ? 3 : 1;
    im.px.resize(static_cast<usize>(W) * static_cast<usize>(H) * im.comps);
    for (s64 r = 0; r < H; ++r)
        for (s64 c = 0; c < W; ++c) {
            const f32 v = slice_at(raw, a, idx, r, c);
            const f32 gn = std::clamp((v - lo) * inv, 0.0f, 1.0f);
            const u8 gray = static_cast<u8>(gn * 255.0f + 0.5f);
            if (!prob) { im.at(static_cast<int>(r), static_cast<int>(c), 0) = gray; continue; }
            f32 p = std::clamp(slice_at(*prob, a, idx, r, c), 0.0f, 1.0f);
            if (p < o.overlay_thresh) p = 0.0f;
            const f32 al = o.alpha * p;
            for (int ch = 0; ch < 3; ++ch)
                im.at(static_cast<int>(r), static_cast<int>(c), ch) =
                    static_cast<u8>(gray + (static_cast<f32>(o.tint[static_cast<usize>(ch)]) - gray) * al);
        }
    return im;
}

namespace detail {
// Load a volume from .fxvol or .nrrd.
inline Expected<Volume<f32>> load_vol(const std::string& p) {
    if (p.size() > 6 && p.substr(p.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::open(p);
        if (!a) return std::unexpected(a.error());
        return a->read_volume();
    }
    return read_nrrd(p);
}
inline void auto_window(VolumeView<const f32> v, f32& lo, f32& hi) {
    if (hi > lo) return;
    f32 mn = v.data()[0], mx = v.data()[0];
    const s64 n = v.dims().count();
    for (s64 i = 0; i < n; ++i) { const f32 x = v.data()[i]; mn = std::min(mn, x); mx = std::max(mx, x); }
    lo = mn; hi = (mx > mn) ? mx : mn + 1.0f;
}
// True if ffmpeg can init h264_nvenc on this box (probe a tiny encode to null).
inline bool nvenc_ok() {
    return std::system("ffmpeg -hide_banner -loglevel error -f lavfi -i color=c=black:s=256x256:d=0.1 "
                       "-c:v h264_nvenc -f null - >/dev/null 2>&1") == 0;
}
}  // namespace detail

// key=value option lookup over the trailing args.
inline std::string opt_get(std::span<const std::string_view> args, std::string_view key, std::string def) {
    for (auto a : args) {
        const auto eq = a.find('=');
        if (eq != std::string_view::npos && a.substr(0, eq) == key) return std::string(a.substr(eq + 1));
    }
    return def;
}

// `fenix slice <vol> <axis> <index> <out.jpg> [overlay=<pred>] [min= max= alpha= quality=]`
inline Expected<int> slice_cmd(std::span<const std::string_view> args, Context&) {
    if (args.size() < 4) {
        log(LogLevel::error, "usage: fenix slice <vol.nrrd|.fxvol> <axis z|y|x> <index> <out.jpg> "
                             "[overlay=<pred>] [min=v] [max=v] [alpha=0.6] [quality=90]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto ax = parse_axis(args[1]);
    if (!ax) return err(Errc::invalid_argument, "axis must be z|y|x");
    auto raw = detail::load_vol(std::string(args[0]));
    if (!raw) return std::unexpected(raw.error());
    auto idx_r = parse_num<s64>(args[2]);
    if (!idx_r) return std::unexpected(idx_r.error());
    const s64 idx = *idx_r;
    const SliceGeom sg = slice_geom(raw->dims(), *ax);
    if (idx < 0 || idx >= sg.frames)
        return err(Errc::invalid_argument, "slice index " + std::to_string(idx) + " out of range [0, " +
                                            std::to_string(sg.frames) + ")");
    const std::string out(args[3]);

    SliceOpts o;
    auto pf = [](std::string_view s, f32 def) -> Expected<f32> {
        return s.empty() ? Expected<f32>(def) : parse_num<f32>(s);
    };
    auto vmin_r = pf(opt_get(args, "min", "0"), 0.0f);
    if (!vmin_r) return std::unexpected(vmin_r.error());
    o.vmin = *vmin_r;
    auto vmax_r = pf(opt_get(args, "max", "0"), 0.0f);
    if (!vmax_r) return std::unexpected(vmax_r.error());
    o.vmax = *vmax_r;
    auto alpha_r = pf(opt_get(args, "alpha", "0.6"), 0.6f);
    if (!alpha_r) return std::unexpected(alpha_r.error());
    o.alpha = *alpha_r;
    o.tint = parse_color(opt_get(args, "color", "red"));
    auto thresh_r = pf(opt_get(args, "thresh", "0"), 0.0f);
    if (!thresh_r) return std::unexpected(thresh_r.error());
    o.overlay_thresh = *thresh_r;
    detail::auto_window(raw->view(), o.vmin, o.vmax);

    std::optional<Volume<f32>> prob;
    const std::string ov = opt_get(args, "overlay", "");
    if (!ov.empty()) {
        auto p = detail::load_vol(ov);
        if (!p) return std::unexpected(p.error());
        if (!(p->dims() == raw->dims())) return err(Errc::invalid_argument, "overlay dims != volume dims");
        prob = std::move(*p);
    }
    const VolumeView<const f32> pv = prob ? prob->view() : VolumeView<const f32>{};
    Image im = build_slice(raw->view(), *ax, idx, o, prob ? &pv : nullptr);
    auto quality_r = parse_num<int>(opt_get(args, "quality", "90"));
    if (!quality_r) return std::unexpected(quality_r.error());
    if (auto w = write_jpeg(out, im, *quality_r); !w) return std::unexpected(w.error());
    log(LogLevel::info, "slice: {} axis {} idx {} -> {} ({}x{}{})", args[0], args[1], idx, out, im.w, im.h,
        prob ? " +overlay" : "");
    return 0;
}

// `fenix video <vol> <axis> <out.mp4> [overlay=<pred>] [fps=30] [step=1] [min= max= alpha= enc=auto]`
inline Expected<int> video_cmd(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3) {
        log(LogLevel::error, "usage: fenix video <vol.nrrd|.fxvol> <axis z|y|x> <out.mp4> "
                             "[overlay=<pred>] [fps=30] [step=1] [reverse=1] [min= max= alpha=0.6] [enc=auto|nvenc|x264]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto ax = parse_axis(args[1]);
    if (!ax) return err(Errc::invalid_argument, "axis must be z|y|x");
    auto raw = detail::load_vol(std::string(args[0]));
    if (!raw) return std::unexpected(raw.error());
    const std::string out(args[2]);

    SliceOpts o;
    auto pf = [](std::string_view s, f32 def) -> Expected<f32> {
        return s.empty() ? Expected<f32>(def) : parse_num<f32>(s);
    };
    auto vmin_r = pf(opt_get(args, "min", "0"), 0.0f);
    if (!vmin_r) return std::unexpected(vmin_r.error());
    o.vmin = *vmin_r;
    auto vmax_r = pf(opt_get(args, "max", "0"), 0.0f);
    if (!vmax_r) return std::unexpected(vmax_r.error());
    o.vmax = *vmax_r;
    auto alpha_r = pf(opt_get(args, "alpha", "0.6"), 0.6f);
    if (!alpha_r) return std::unexpected(alpha_r.error());
    o.alpha = *alpha_r;
    o.tint = parse_color(opt_get(args, "color", "red"));
    auto thresh_r = pf(opt_get(args, "thresh", "0"), 0.0f);
    if (!thresh_r) return std::unexpected(thresh_r.error());
    o.overlay_thresh = *thresh_r;
    detail::auto_window(raw->view(), o.vmin, o.vmax);
    auto fps_r = parse_num<int>(opt_get(args, "fps", "30"));
    if (!fps_r) return std::unexpected(fps_r.error());
    const int fps = *fps_r;
    auto step_r = parse_num<s64>(opt_get(args, "step", "1"));
    if (!step_r) return std::unexpected(step_r.error());
    const s64 step = std::max<s64>(1, *step_r);
    const bool reverse = opt_get(args, "reverse", "0") != "0";

    std::optional<Volume<f32>> prob;
    const std::string ov = opt_get(args, "overlay", "");
    if (!ov.empty()) {
        auto p = detail::load_vol(ov);
        if (!p) return std::unexpected(p.error());
        if (!(p->dims() == raw->dims())) return err(Errc::invalid_argument, "overlay dims != volume dims");
        prob = std::move(*p);
    }
    const VolumeView<const f32> pv = prob ? prob->view() : VolumeView<const f32>{};
    const bool color = prob.has_value();

    const SliceGeom g = slice_geom(raw->dims(), *ax);
    const s64 W = g.W & ~s64{1}, H = g.H & ~s64{1};

    // Encoder: NVENC (GPU) if it initializes here, else libx264 ultrafast. mp4/yuv420p/+faststart
    // is broadly playable (VLC, browsers). Frames are piped in as raw rgb24 (or gray8).
    std::string enc = opt_get(args, "enc", "auto");
    if (enc == "auto") enc = detail::nvenc_ok() ? "nvenc" : "x264";
    const std::string vcodec = enc == "nvenc" ? "h264_nvenc -preset p4 -tune ll"
                                              : "libx264 -preset ultrafast -crf 23";
    const std::string pixfmt_in = color ? "rgb24" : "gray";
    char cmd[1024];
    std::snprintf(cmd, sizeof cmd,
                  "ffmpeg -y -loglevel error -f rawvideo -pix_fmt %s -video_size %lldx%lld "
                  "-framerate %d -i - -c:v %s -pix_fmt yuv420p -movflags +faststart \"%s\"",
                  pixfmt_in.c_str(), static_cast<long long>(W), static_cast<long long>(H), fps,
                  vcodec.c_str(), out.c_str());
    // Frame index order (forward 0..N or reverse N..0, stepped).
    std::vector<s64> order;
    for (s64 idx = 0; idx < g.frames; idx += step) order.push_back(idx);
    if (reverse) std::reverse(order.begin(), order.end());

    log(LogLevel::info, "video: {} axis {} -> {} ({}x{}, {} frames, enc={}{})", args[0], args[1], out, W, H,
        order.size(), enc, reverse ? ", reverse" : "");

    std::FILE* pipe = ::popen(cmd, "w");
    if (!pipe) return err(Errc::io_error, "video: failed to launch ffmpeg (is it installed?)");

    // If ffmpeg dies early (bad codec, disk full), the next fwrite to the now-closed pipe raises
    // SIGPIPE — default action kills this whole process, so the pclose()/Errc::io_error handling
    // below never runs. Ignore it here (scoped to this call) so a broken pipe surfaces as EPIPE on
    // fwrite instead, which we check explicitly and turn into a clean Error.
    void (*old_sigpipe)(int) = std::signal(SIGPIPE, SIG_IGN);

    s64 n = 0;
    bool write_failed = false;
    for (s64 idx : order) {
        Image im = build_slice(raw->view(), *ax, idx, o, color ? &pv : nullptr);
        if (std::fwrite(im.px.data(), 1, im.px.size(), pipe) != im.px.size()) { write_failed = true; break; }
        if (++n % 100 == 0) log(LogLevel::info, "video: {} frames", n);
    }
    const int rc = ::pclose(pipe);
    std::signal(SIGPIPE, old_sigpipe);
    if (write_failed) return err(Errc::io_error, "video: write to ffmpeg pipe failed (ffmpeg exited early?)");
    if (rc != 0) return err(Errc::io_error, "video: ffmpeg exited with code " + std::to_string(rc));
    log(LogLevel::info, "video: wrote {} ({} frames)", out, n);
    return 0;
}

}  // namespace fenix::io

namespace {
[[maybe_unused]] const int fenix_stage_slice =
    ::fenix::register_stage(::fenix::Stage{"slice", "export a 2D slice (any axis) as JPEG (+red overlay)",
                                           ::fenix::io::slice_cmd});
[[maybe_unused]] const int fenix_stage_video =
    ::fenix::register_stage(::fenix::Stage{"video", "export an H.264 .mp4 scrubbing any axis (NVENC/x264, +overlay)",
                                           ::fenix::io::video_cmd});
}  // namespace
