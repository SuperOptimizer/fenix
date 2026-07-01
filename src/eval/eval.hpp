// eval/eval.hpp — the `eval` stage: score a predicted segmentation/surface against a ground-truth (or
// pseudo-GT teacher) volume. Wires the metric primitives (score.hpp / nsd.hpp / metrics.hpp) to a CLI so
// distillation / accel / TTA changes can be MEASURED. Header-only; self-registering stage.
#pragma once

#include "core/core.hpp"

#include "eval/deformation.hpp"
#include "eval/mesh_quality.hpp"
#include "eval/metrics.hpp"
#include "eval/nsd.hpp"
#include "eval/score.hpp"

#include "codec/archive.hpp"
#include "io/nrrd.hpp"

#include <algorithm>
#include <charconv>
#include <string>
#include <string_view>

namespace fenix::eval {

namespace detail {

// Load a volume (`.fxvol` archive or `.nrrd`) as f32 (LOD 0).
inline Expected<Volume<f32>> load_f32(const std::string& path) {
    if (path.size() > 6 && path.substr(path.size() - 6) == ".fxvol") {
        auto a = codec::VolumeArchive::open(path);
        if (!a) return std::unexpected(a.error());
        return a->read_volume();
    }
    return io::read_nrrd(path);
}

// Threshold an f32 volume into a binary u8 mask (v >= thr → 1). `thr` is on the data's own scale.
inline Volume<u8> threshold(VolumeView<const f32> v, f32 thr) {
    Volume<u8> out = Volume<u8>::zeros(v.dims());
    auto of = out.view().flat();
    const auto f = v.flat();
    for (s64 i = 0; i < v.size(); ++i) of[static_cast<usize>(i)] = f[static_cast<usize>(i)] >= thr ? u8{1} : u8{0};
    return out;
}

// Peak value (to auto-scale the default threshold: 0..1 softmax probs vs the codec's 0..255 round-trip).
inline f32 peak(VolumeView<const f32> v) {
    f32 mx = 0.0f;
    const auto f = v.flat();
    for (s64 i = 0; i < v.size(); ++i) mx = std::max(mx, f[static_cast<usize>(i)]);
    return mx;
}

inline f64 parse_f(std::string_view s, f64 def) {
    f64 v = def;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

}  // namespace detail

// `fenix eval <pred.fxvol|.nrrd> <gt.fxvol|.nrrd> [--tau T] [--thresh X] [--gt-thresh X] [--json]`
// Threshold both volumes to binary masks and print the Kaggle composite (TopoScore/SurfaceDice@τ/VOI) +
// Dice/IoU. `--thresh`/`--gt-thresh` are on the data's own scale; if omitted, default to 0.5·peak (works
// whether the volume is 0..1 or the codec's 0..255). `--json` emits one machine-readable line for baseline
// storage / regression gates.
inline Expected<int> run(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix eval <pred> <gt> [--tau T] [--thresh X] [--gt-thresh X] [--json]");
        return err(Errc::invalid_argument, "need <pred> <gt>");
    }
    const std::string pred_path(args[0]), gt_path(args[1]);
    f32 tau = 2.0f;
    f64 pred_thr = -1.0, gt_thr = -1.0;  // <0 ⇒ auto (0.5·peak)
    bool json = false;
    for (usize i = 2; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--tau" && i + 1 < args.size()) tau = static_cast<f32>(detail::parse_f(args[++i], 2.0));
        else if (a == "--thresh" && i + 1 < args.size()) pred_thr = detail::parse_f(args[++i], 0.5);
        else if (a == "--gt-thresh" && i + 1 < args.size()) gt_thr = detail::parse_f(args[++i], 0.5);
        else if (a == "--json") json = true;
    }

    auto pv = detail::load_f32(pred_path);
    if (!pv) return std::unexpected(pv.error());
    auto gv = detail::load_f32(gt_path);
    if (!gv) return std::unexpected(gv.error());
    if (!(pv->dims() == gv->dims())) return err(Errc::invalid_argument, "pred/gt dims differ");

    if (pred_thr < 0.0) pred_thr = 0.5 * static_cast<f64>(detail::peak(pv->view()));
    if (gt_thr < 0.0) gt_thr = 0.5 * static_cast<f64>(detail::peak(gv->view()));
    if (pred_thr <= 0.0) pred_thr = 0.5;
    if (gt_thr <= 0.0) gt_thr = 0.5;

    const Volume<u8> pm = detail::threshold(pv->view(), static_cast<f32>(pred_thr));
    const Volume<u8> gm = detail::threshold(gv->view(), static_cast<f32>(gt_thr));
    const VolumeView<const u8> pmv = pm.view(), gmv = gm.view();

    const Score s = official_score(pmv, gmv, tau);
    const f64 d = dice(pmv, gmv);
    const f64 j = iou(pmv, gmv);

    const Extent3 dd = pv->dims();
    if (json) {
        log(LogLevel::info,
            R"({{"pred":"{}","gt":"{}","dims":[{},{},{}],"tau":{:.3g},"pred_thresh":{:.4g},"gt_thresh":{:.4g},)"
            R"("official":{:.5f},"surface_dice":{:.5f},"voi_score":{:.5f},"topo_score":{:.5f},"dice":{:.5f},"iou":{:.5f}}})",
            pred_path, gt_path, dd.z, dd.y, dd.x, tau, pred_thr, gt_thr, s.total, s.surface_dice, s.voi_score,
            s.topo_score, d, j);
    } else {
        log(LogLevel::info, "eval {} vs {} ({}x{}x{} ZYX)", pred_path, gt_path, dd.z, dd.y, dd.x);
        log(LogLevel::info, "  thresholds: pred>={:.4g}  gt>={:.4g}  tau={:.3g}", pred_thr, gt_thr, tau);
        log(LogLevel::info, "  official   = {:.5f}   (0.30*Topo + 0.35*SurfDice + 0.35*VOI)", s.total);
        log(LogLevel::info, "  SurfaceDice@{:.3g} = {:.5f}", tau, s.surface_dice);
        log(LogLevel::info, "  VOI_score        = {:.5f}", s.voi_score);
        log(LogLevel::info, "  TopoScore        = {:.5f}", s.topo_score);
        log(LogLevel::info, "  Dice / IoU       = {:.5f} / {:.5f}", d, j);
    }
    return 0;
}

}  // namespace fenix::eval

FENIX_REGISTER_STAGE(eval, "score a prediction vs ground-truth (composite + surface-dice/voi/topo/dice)",
                     ::fenix::eval::run)
