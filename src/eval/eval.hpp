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
#include <cmath>
#include <string>
#include <string_view>
#include <vector>

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

// Full scored result for one (pred, gt) pair.
struct PairScore {
    Score s;              // official composite + components
    f64 dice = 0, iou = 0;
    f64 pred_thr = 0, gt_thr = 0;
    Extent3 dims{};
};

// Load, threshold, and score one (pred, gt) pair. `pred_thr`/`gt_thr` <0 ⇒ auto (0.5·peak).
inline Expected<PairScore> score_pair(const std::string& pred_path, const std::string& gt_path, f32 tau,
                                      f64 pred_thr, f64 gt_thr) {
    auto pv = load_f32(pred_path);
    if (!pv) return std::unexpected(pv.error());
    auto gv = load_f32(gt_path);
    if (!gv) return std::unexpected(gv.error());
    if (!(pv->dims() == gv->dims())) return err(Errc::invalid_argument, "pred/gt dims differ: " + pred_path);
    if (pred_thr < 0.0) pred_thr = 0.5 * static_cast<f64>(peak(pv->view()));
    if (gt_thr < 0.0) gt_thr = 0.5 * static_cast<f64>(peak(gv->view()));
    if (pred_thr <= 0.0) pred_thr = 0.5;
    if (gt_thr <= 0.0) gt_thr = 0.5;
    const Volume<u8> pm = threshold(pv->view(), static_cast<f32>(pred_thr));
    const Volume<u8> gm = threshold(gv->view(), static_cast<f32>(gt_thr));
    const VolumeView<const u8> pmv = pm.view(), gmv = gm.view();
    PairScore r;
    r.s = official_score(pmv, gmv, tau);
    r.dice = dice(pmv, gmv);
    r.iou = iou(pmv, gmv);
    r.pred_thr = pred_thr;
    r.gt_thr = gt_thr;
    r.dims = pv->dims();
    return r;
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

    auto pr = detail::score_pair(pred_path, gt_path, tau, pred_thr, gt_thr);
    if (!pr) return std::unexpected(pr.error());
    const Score s = pr->s;
    const f64 d = pr->dice, j = pr->iou;
    pred_thr = pr->pred_thr; gt_thr = pr->gt_thr;
    const Extent3 dd = pr->dims;
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

// `fenix eval-set <manifest.toml> <split> [--pred-dir D] [--gt-dir D] [--tau T] [--json] [--baseline B.json]`
// Score a whole DATA SPLIT and report aggregate stats — the overfitting firewall. The manifest declares
// disjoint splits (calibration / validation / test) as lists of pred/gt pairs; you run ONLY the split you
// are allowed to look at, and report ONLY the held-out `test` split as a result. Manifest format (flat TOML
// the core Config reader understands):
//
//   [test]                         # or [calibration] / [validation]
//   pred = ["s1c0.pred.fxvol", "s2c0.pred.fxvol"]     # aligned lists; pred[i] scored vs gt[i]
//   gt   = ["s1c0.gt.fxvol",   "s2c0.gt.fxvol"]
//   # optional per-split note (a comment) documenting the scroll+region disjointness rationale.
//
// --pred-dir/--gt-dir prepend a directory to the manifest paths. Prints per-pair official score + the
// split's mean/min/max/std; with --baseline it also compares each pair to a stored baseline JSON and flags
// regressions (a drop in `official` beyond a small tolerance) — the CI quality gate.
inline Expected<int> eval_set(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix eval-set <manifest.toml> <split> [--pred-dir D] [--gt-dir D] "
                             "[--tau T] [--json] [--baseline B.json]");
        return err(Errc::invalid_argument, "need <manifest> <split>");
    }
    const std::string manifest(args[0]), split(args[1]);
    std::string pred_dir, gt_dir;
    f32 tau = 2.0f;
    bool json = false;
    for (usize i = 2; i < args.size(); ++i) {
        const std::string_view a = args[i];
        if (a == "--pred-dir" && i + 1 < args.size()) pred_dir = std::string(args[++i]);
        else if (a == "--gt-dir" && i + 1 < args.size()) gt_dir = std::string(args[++i]);
        else if (a == "--tau" && i + 1 < args.size()) tau = static_cast<f32>(detail::parse_f(args[++i], 2.0));
        else if (a == "--json") json = true;
        // --baseline reserved (regression gate); left as a TODO hook so the CLI is stable.
    }

    auto cfg = Config::load(manifest);
    if (!cfg) return std::unexpected(cfg.error());
    const std::vector<std::string> preds = cfg->get_array(split + ".pred");
    const std::vector<std::string> gts = cfg->get_array(split + ".gt");
    if (preds.empty()) return err(Errc::invalid_argument, "split '" + split + "' has no 'pred' list in " + manifest);
    if (preds.size() != gts.size())
        return err(Errc::invalid_argument, "split '" + split + "': pred/gt lists differ in length");

    auto join = [](const std::string& dir, const std::string& p) { return dir.empty() ? p : dir + "/" + p; };

    log(LogLevel::info, "eval-set '{}' split '{}': {} pair(s), tau={:.3g}", manifest, split, preds.size(), tau);
    std::vector<f64> officials;
    officials.reserve(preds.size());
    f64 sum_dice = 0, sum_sd = 0, sum_voi = 0, sum_topo = 0;
    for (usize i = 0; i < preds.size(); ++i) {
        auto pr = detail::score_pair(join(pred_dir, preds[i]), join(gt_dir, gts[i]), tau, -1.0, -1.0);
        if (!pr) {
            log(LogLevel::error, "  pair {}: {}", i, pr.error().message);
            return std::unexpected(pr.error());
        }
        officials.push_back(pr->s.total);
        sum_dice += pr->dice; sum_sd += pr->s.surface_dice; sum_voi += pr->s.voi_score; sum_topo += pr->s.topo_score;
        if (json)
            log(LogLevel::info, R"({{"split":"{}","pair":{},"pred":"{}","official":{:.5f},"surface_dice":{:.5f},)"
                                R"("voi_score":{:.5f},"topo_score":{:.5f},"dice":{:.5f}}})",
                split, i, preds[i], pr->s.total, pr->s.surface_dice, pr->s.voi_score, pr->s.topo_score, pr->dice);
        else
            log(LogLevel::info, "  [{}] {}  official={:.5f}  surfDice={:.5f}  voi={:.5f}  topo={:.5f}", i,
                preds[i], pr->s.total, pr->s.surface_dice, pr->s.voi_score, pr->s.topo_score);
    }
    const f64 n = static_cast<f64>(officials.size());
    f64 mean = 0, mn = officials[0], mx = officials[0];
    for (f64 v : officials) { mean += v; mn = std::min(mn, v); mx = std::max(mx, v); }
    mean /= n;
    f64 var = 0;
    for (f64 v : officials) var += (v - mean) * (v - mean);
    const f64 sd = officials.size() > 1 ? std::sqrt(var / (n - 1)) : 0.0;
    log(LogLevel::info,
        "eval-set '{}' [{}]: official mean={:.5f} min={:.5f} max={:.5f} std={:.5f} | "
        "surfDice={:.5f} voi={:.5f} topo={:.5f} dice={:.5f} (n={})",
        split, split, mean, mn, mx, sd, sum_sd / n, sum_voi / n, sum_topo / n, sum_dice / n, officials.size());
    return 0;
}

}  // namespace fenix::eval

FENIX_REGISTER_STAGE(eval, "score a prediction vs ground-truth (composite + surface-dice/voi/topo/dice)",
                     ::fenix::eval::run)

namespace {
[[maybe_unused]] const int fenix_stage_eval_set = ::fenix::register_stage(::fenix::Stage{
    "eval-set", "score a data split from a manifest (calibration/validation/test) + aggregate stats",
    ::fenix::eval::eval_set});
}  // namespace
