// segment/stream_grow.hpp — the STREAMING FRONTIER grower: one sheet grown across block
// boundaries with data fetched on demand around the live frontier. The windowed twin of
// grow_surface's frontier loop (same accept gates: snap/spacing/CT-valley/injectivity/
// winding); geometry lives in WORLD coordinates, the data fields carry a window origin
// (DataField/NormalField::org), and a frontier cell landing OUTSIDE the active window is
// PAUSED, not killed — the driver re-centres the window on the biggest paused cluster,
// fetches the next block (type-erased fetch fn -> CachedVolume/zarr), and resumes. This
// is what removes the 512-block ceiling on segment length (the GP-length path).
// v1 scope: u8 fields; final DATA-dependent polish passes (global ARAP/fill) are skipped
// (light per-window ARAP runs during growth via fit_every); the model winding_filter +
// data-free cleanups run globally at the end. See segment/CLAUDE.md.
#pragma once

#include "core/core.hpp"
#include "segment/grow.hpp"

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

namespace fenix::segment {

// fetch(world_lo, dims, pred_out, ct_out) -> false on unrecoverable fetch failure.
// ct_out may be left empty (no CT term). Fetch failure must NEVER be treated as air —
// the driver aborts the window (absent vs failed are distinct; core rule).
using WindowFetch = std::function<bool(Vec3f, Extent3, Volume<u8>&, Volume<u8>&)>;

struct StreamGrowStats {
    s64 windows = 0, cells = 0, paused_final = 0;
};

class StreamGrower {
public:
    StreamGrower(GrowParams p, Vec3f full_lo, Vec3f full_hi, Extent3 window, s64 sheet_id = 0)
        : p_(std::move(p)), full_lo_(full_lo), full_hi_(full_hi), win_(window), sheet_id_(sheet_id),
          S_(p_.grid, p_.grid) {
        const usize NG = static_cast<usize>(p_.grid) * static_cast<usize>(p_.grid);
        dead_.assign(NG, 0);
        queued_.assign(NG, 0);
        bdepth_.assign(NG, 0);
        Tu_.assign(NG, Vec3f{0, 0, 0});
        Tv_.assign(NG, Vec3f{0, 0, 0});
        if (p_.winding_tol > 0 && p_.winding_fn) wcell_.assign(NG, 1e30f);
        occ_.reserve(1 << 16);
    }

    // Grow from `seed` (world), streaming windows via `fetch`, until no frontier remains
    // anywhere or `max_windows` is hit. Returns the surface in world coords.
    Expected<Surface> run(Vec3f seed, const WindowFetch& fetch, int max_windows, StreamGrowStats* stats) {
        Vec3f center = seed;
        bool first = true;
        int wi = 0;
        for (; wi < max_windows; ++wi) {
            // window box centred on target, clamped to the full volume
            Vec3f wlo{center.z - static_cast<f32>(win_.z) / 2, center.y - static_cast<f32>(win_.y) / 2,
                      center.x - static_cast<f32>(win_.x) / 2};
            wlo = Vec3f{std::clamp(wlo.z, full_lo_.z, std::max(full_lo_.z, full_hi_.z - static_cast<f32>(win_.z))),
                        std::clamp(wlo.y, full_lo_.y, std::max(full_lo_.y, full_hi_.y - static_cast<f32>(win_.y))),
                        std::clamp(wlo.x, full_lo_.x, std::max(full_lo_.x, full_hi_.x - static_cast<f32>(win_.x)))};
            wlo = Vec3f{std::floor(wlo.z), std::floor(wlo.y), std::floor(wlo.x)};
            Volume<u8> pred, ct;
            if (!fetch(wlo, win_, pred, ct))
                return err(Errc::io_error, "stream-grow: window fetch failed (never treating as air)");
            DataField<u8> fld{pred.view(), ct.dims().count() > 0 ? ct.view() : VolumeView<const u8>{},
                              p_.surf_thresh, p_.ct_thresh, p_.ct_weight, p_.ct_ds, p_.ct_skip};
            fld.org = wlo;
            NormalField nf = compute_normal_field<u8>(pred.view(), 8);
            nf.org = wlo;
            GrowParams wp = p_;
            if (ct.dims().count() == 0) wp.ct_barrier = 0;

            if (first) {
                if (!seed_patch(fld, nf, seed, wlo)) return err(Errc::invalid_argument, "stream-grow: seed failed to snap");
                first = false;
            } else {
                requeue_paused_in(wlo);
            }
            const s64 placed = run_window(fld, nf, wp, wlo);
            log(LogLevel::info, "stream-grow: window {} at ({:.0f},{:.0f},{:.0f}) placed {} (total {}, paused {})",
                wi + 1, wlo.z, wlo.y, wlo.x, placed, S_.valid_count(), paused_.size());
            if (paused_.empty()) { ++wi; break; }
            center = pick_next_center(wlo);
        }
        // data-free global finish: winding filter + de-fragment
        if (p_.winding_tol > 0 && p_.winding_fn)
            winding_filter(S_, p_.winding_fn, p_.winding_tol, p_.winding_jump, p_.grid / 2, p_.grid / 2);
        detail::remove_slivers(S_, p_);
        detail::keep_large_components(S_, 200);
        if (stats) {
            stats->windows = wi;
            stats->cells = S_.valid_count();
            stats->paused_final = static_cast<s64>(paused_.size());
        }
        return std::move(S_);
    }

private:
    GrowParams p_;
    Vec3f full_lo_, full_hi_;
    Extent3 win_;
    s64 sheet_id_;
    Surface S_;
    std::vector<u8> dead_, queued_, bdepth_;
    std::vector<Vec3f> Tu_, Tv_;
    std::vector<f32> wcell_;
    f32 wind_ref_ = 0;
    OccMap occ_;
    std::vector<s64> frontier_;
    std::vector<std::pair<s64, Vec3f>> paused_;  // (uv id, predicted world pos at pause time)

    [[nodiscard]] bool in_window(Vec3f c, Vec3f wlo) const {
        const f32 m = p_.snap_radius + 2.0f;
        return c.z >= wlo.z + m && c.y >= wlo.y + m && c.x >= wlo.x + m &&
               c.z < wlo.z + static_cast<f32>(win_.z) - m && c.y < wlo.y + static_cast<f32>(win_.y) - m &&
               c.x < wlo.x + static_cast<f32>(win_.x) - m;
    }
    [[nodiscard]] bool in_full(Vec3f c) const {
        return c.z >= full_lo_.z && c.y >= full_lo_.y && c.x >= full_lo_.x && c.z < full_hi_.z &&
               c.y < full_hi_.y && c.x < full_hi_.x;
    }
    // reachable by SOME window: inside the full volume with the window margin to spare —
    // a cell closer than the margin to the WORLD boundary can never be grown and must
    // die, not pause (paused world-edge cells strand the driver: it burns every window
    // re-centring on a cluster no window can ever contain)
    [[nodiscard]] bool window_reachable(Vec3f c) const {
        const f32 m = p_.snap_radius + 2.0f;
        return c.z >= full_lo_.z + m && c.y >= full_lo_.y + m && c.x >= full_lo_.x + m &&
               c.z < full_hi_.z - m && c.y < full_hi_.y - m && c.x < full_hi_.x - m;
    }

    static std::pair<Vec3f, Vec3f> transport(Vec3f tu, Vec3f tv, Vec3f n) {
        Vec3f u = tu - n * dot(tu, n);
        if (norm(u) < 1e-4f) {
            const Vec3f e = (std::abs(n.x) < 0.9f) ? Vec3f{0, 0, 1} : Vec3f{1, 0, 0};
            u = e - n * dot(e, n);
        }
        u = normalized(u);
        Vec3f v = tv - n * dot(tv, n);
        v = v - u * dot(v, u);
        if (norm(v) < 1e-4f) v = cross(n, u);
        return {u, normalized(v)};
    }

    bool seed_patch(const DataField<u8>& fld, const NormalField& nf, Vec3f seed, Vec3f wlo) {
        const int G = p_.grid, C = G / 2;
        const Vec3f n0 = nf.at(seed);
        auto [sp, sv0, st0] = detail::snap_to_sheet(fld, seed, n0, p_.snap_radius);
        if (sv0 < 1.0f || !in_window(sp, wlo)) return false;
        (void)st0;
        auto [su, sv] = transport(Vec3f{0, 0, 1}, Vec3f{0, 1, 0}, n0);
        S_.set(C, C, sp);
        Tu_[S_.idx(C, C)] = su;
        Tv_[S_.idx(C, C)] = sv;
        for (int dv = -1; dv <= 1; ++dv)
            for (int du = -1; du <= 1; ++du) {
                if (du == 0 && dv == 0) continue;
                const Vec3f c = sp + su * (static_cast<f32>(du) * p_.step) + sv * (static_cast<f32>(dv) * p_.step);
                if (!in_window(c, wlo)) continue;
                auto [q, vv, tt] = detail::snap_to_sheet(fld, c, nf.at(c), p_.snap_radius);
                (void)tt;
                if (vv >= 1.0f && in_window(q, wlo)) {
                    S_.set(C + du, C + dv, q);
                    auto [fu, fv] = transport(su, sv, nf.at(q));
                    Tu_[S_.idx(C + du, C + dv)] = fu;
                    Tv_[S_.idx(C + du, C + dv)] = fv;
                }
            }
        if (p_.winding_tol > 0 && p_.winding_fn) {
            wind_ref_ = p_.winding_fn(S_.at(C, C));
            for (int dv = -1; dv <= 1; ++dv)
                for (int du = -1; du <= 1; ++du)
                    if (S_.is_valid(C + du, C + dv)) {
                        const f32 w = p_.winding_fn(S_.at(C + du, C + dv));
                        wcell_[S_.idx(C + du, C + dv)] = w + std::round(wind_ref_ - w);
                    }
        }
        for (int dv = -1; dv <= 1; ++dv)
            for (int du = -1; du <= 1; ++du)
                if (S_.is_valid(C + du, C + dv)) claim(S_.at(C + du, C + dv), C + du, C + dv);
        frontier_.clear();
        for (int dv = -1; dv <= 1; ++dv)
            for (int du = -1; du <= 1; ++du)
                if (S_.is_valid(C + du, C + dv)) enqueue_neighbours(C + du, C + dv);
        return true;
    }

    // --- occupancy (injectivity guard), same binning as grow_surface ---
    static constexpr s64 kBS = 1 << 20;
    static constexpr s64 kShift = static_cast<s64>(1) << 32;
    [[nodiscard]] s64 binkey(Vec3f c, int oz, int oy, int ox) const {
        const f32 ib = 1.0f / p_.bin_size;
        return (static_cast<s64>(c.z * ib) + oz) * kBS * kBS + (static_cast<s64>(c.y * ib) + oy) * kBS +
               (static_cast<s64>(c.x * ib) + ox);
    }
    [[nodiscard]] bool fold_conflict(Vec3f q, int u, int v) {
        for (int oz = -1; oz <= 1; ++oz)
            for (int oy = -1; oy <= 1; ++oy)
                for (int ox = -1; ox <= 1; ++ox) {
                    const s64* it = occ_.get(binkey(q, oz, oy, ox));
                    if (!it) continue;
                    const s64 cell = *it % kShift;
                    const int ou = static_cast<int>(cell % p_.grid), ov = static_cast<int>(cell / p_.grid);
                    if (std::abs(u - ou) + std::abs(v - ov) > p_.fold_thresh) return true;
                }
        return false;
    }
    void claim(Vec3f q, int u, int v) {
        occ_.set(binkey(q, 0, 0, 0), sheet_id_ * kShift + (static_cast<s64>(v) * p_.grid + u));
    }
    void enqueue_neighbours(int u, int v) {
        static constexpr int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
        for (int k = 0; k < 4; ++k) {
            const int uu = u + du4[k], vv = v + dv4[k];
            if (uu < 1 || vv < 1 || uu >= p_.grid - 1 || vv >= p_.grid - 1) continue;
            const s64 id = static_cast<s64>(vv) * p_.grid + uu;
            if (S_.valid[static_cast<usize>(id)] || dead_[static_cast<usize>(id)] || queued_[static_cast<usize>(id)])
                continue;
            queued_[static_cast<usize>(id)] = 1;
            frontier_.push_back(id);
        }
    }
    void requeue_paused_in(Vec3f wlo) {
        std::vector<std::pair<s64, Vec3f>> keep;
        for (const auto& [id, pos] : paused_) {
            if (!window_reachable(pos)) {
                dead_[static_cast<usize>(id)] = 1;  // no window can ever contain it
            } else if (in_window(pos, wlo)) {
                frontier_.push_back(id);  // still queued_ == 1
            } else {
                keep.push_back({id, pos});
            }
        }
        paused_.swap(keep);
    }
    [[nodiscard]] Vec3f pick_next_center(Vec3f cur_wlo) const {
        // biggest cluster proxy: the paused position farthest INSIDE the full volume,
        // averaged with its 32-vox neighbours (cheap; window overlap does the rest)
        (void)cur_wlo;
        Vec3f best = paused_.front().second;
        s64 best_n = -1;
        for (usize i = 0; i < paused_.size(); i += std::max<usize>(1, paused_.size() / 64)) {
            const Vec3f c = paused_[i].second;
            s64 n = 0;
            Vec3f acc{0, 0, 0};
            for (const auto& [id2, q] : paused_) {
                (void)id2;
                if (std::abs(q.z - c.z) < 32 && std::abs(q.y - c.y) < 32 && std::abs(q.x - c.x) < 32) {
                    acc = acc + q;
                    ++n;
                }
            }
            if (n > best_n) {
                best_n = n;
                best = acc / static_cast<f32>(n);
            }
        }
        return best;
    }

    // one window's frontier growth — grow_surface's loop with world coords + pausing
    s64 run_window(const DataField<u8>& fld, const NormalField& nf, const GrowParams& p, Vec3f wlo) {
        static constexpr int du4[4] = {-1, 1, 0, 0}, dv4[4] = {0, 0, -1, 1};
        const int G = p.grid;
        s64 total_placed = 0;
        std::vector<s64> cur;
        cur.swap(frontier_);
        for (int gen = 0; gen < p.max_gen && !cur.empty(); ++gen) {
            std::sort(cur.begin(), cur.end());
            int placed = 0;
            for (const s64 id : cur) {
                const int u = static_cast<int>(id % G), v = static_cast<int>(id / G);
                Vec3f pred_c{0, 0, 0}, accu{0, 0, 0}, accv{0, 0, 0};
                int np = 0;
                for (int k = 0; k < 4; ++k) {
                    const int nu = u + du4[k], nv = v + dv4[k];
                    if (!S_.is_valid(nu, nv)) continue;
                    const usize ni = S_.idx(nu, nv);
                    const f32 dU = static_cast<f32>(u - nu), dV = static_cast<f32>(v - nv);
                    pred_c = pred_c + (S_.at(nu, nv) + (Tu_[ni] * dU + Tv_[ni] * dV) * p.step);
                    accu = accu + Tu_[ni];
                    accv = accv + Tv_[ni];
                    ++np;
                }
                if (np == 0) { queued_[static_cast<usize>(id)] = 0; continue; }
                const Vec3f c = pred_c / static_cast<f32>(np);
                if (!window_reachable(c)) { dead_[static_cast<usize>(id)] = 1; continue; }
                if (!in_window(c, wlo)) { paused_.push_back({id, c}); continue; }  // resume in a later window
                const Vec3f nrm = nf.at(c);
                auto [q, val, tt] = detail::snap_to_sheet(fld, c, nrm, p.snap_radius);
                const bool inq = in_window(q, wlo);
                const bool snapped = inq && val >= 1.0f && std::abs(tt) <= p.snap_radius - 0.51f;
                u8 bd = 0;
                Vec3f place = q;
                if (!snapped) {
                    if (p.max_bridge <= 0) { dead_[static_cast<usize>(id)] = 1; continue; }
                    int mind = 255;
                    for (int k = 0; k < 4; ++k) {
                        const int uu = u + du4[k], vv = v + dv4[k];
                        if (S_.is_valid(uu, vv)) mind = std::min(mind, static_cast<int>(bdepth_[S_.idx(uu, vv)]));
                    }
                    bd = static_cast<u8>(std::min(254, mind + 1));
                    if (bd > p.max_bridge) { dead_[static_cast<usize>(id)] = 1; continue; }
                    const f32 w = (p.soft_gate && inq) ? std::clamp((val - 0.5f) * 2.0f, 0.0f, 1.0f) : 0.0f;
                    place = c + (q - c) * w;
                }
                if (!window_reachable(place)) { dead_[static_cast<usize>(id)] = 1; continue; }
                if (!in_window(place, wlo)) { paused_.push_back({id, place}); continue; }
                bool ok = true;
                for (int k = 0; k < 4; ++k) {
                    const int uu = u + du4[k], vv = v + dv4[k];
                    if (S_.is_valid(uu, vv)) {
                        const f32 d = norm(place - S_.at(uu, vv));
                        if (d > 2.5f * p.step || d < 0.5f * p.step) ok = false;
                    }
                }
                if (!ok) { dead_[static_cast<usize>(id)] = 1; continue; }
                if (p.ct_barrier > 0.0f && fld.has_ct()) {
                    bool jumped = false;
                    for (int k = 0; k < 4 && !jumped; ++k) {
                        const int uu = u + du4[k], vv = v + dv4[k];
                        if (S_.is_valid(uu, vv))
                            jumped = crosses_valley(fld.ct, fld.ct_coord(S_.at(uu, vv)), fld.ct_coord(place),
                                                    p.ct_barrier, 0.5f);
                    }
                    if (jumped) { dead_[static_cast<usize>(id)] = 1; continue; }
                }
                if (fold_conflict(place, u, v)) { dead_[static_cast<usize>(id)] = 1; continue; }
                if (p.winding_tol > 0 && p.winding_fn) {
                    const f32 w = p.winding_fn(place);
                    if (!(std::abs(w) < 1e30f)) { dead_[static_cast<usize>(id)] = 1; continue; }
                    f32 wp = 0;
                    int nwp = 0;
                    for (int k = 0; k < 4; ++k) {
                        const int uu = u + du4[k], vv = v + dv4[k];
                        if (S_.is_valid(uu, vv) && wcell_[S_.idx(uu, vv)] < 1e29f) {
                            wp += wcell_[S_.idx(uu, vv)];
                            ++nwp;
                        }
                    }
                    const f32 pref = nwp > 0 ? wp / static_cast<f32>(nwp) : wind_ref_;
                    const f32 res = w + std::round(pref - w);
                    if (std::abs(res - pref) > p.winding_jump || std::abs(res - wind_ref_) > p.winding_tol) {
                        dead_[static_cast<usize>(id)] = 1;
                        continue;
                    }
                    wcell_[static_cast<usize>(id)] = res;
                }
                S_.set(u, v, place);
                claim(place, u, v);
                bdepth_[static_cast<usize>(id)] = bd;
                auto [fu, fv] = transport(normalized(accu), normalized(accv), nf.at(place));
                Tu_[S_.idx(u, v)] = fu;
                Tv_[S_.idx(u, v)] = fv;
                ++placed;
                ++total_placed;
                enqueue_neighbours(u, v);
            }
            cur.swap(frontier_);
            frontier_.clear();
            if (placed == 0 && cur.empty()) break;
            if (placed == 0 && gen > 0) break;  // only paused cells remain reachable
        }
        // anything left un-run stays queued for the next window, at its parents' mean
        // position (a real, re-window-able location — a sentinel would strand it forever)
        static constexpr int du4b[4] = {-1, 1, 0, 0}, dv4b[4] = {0, 0, -1, 1};
        for (const s64 id : cur) {
            const int u = static_cast<int>(id % G), v = static_cast<int>(id / G);
            Vec3f acc{0, 0, 0};
            int n = 0;
            for (int k = 0; k < 4; ++k)
                if (S_.is_valid(u + du4b[k], v + dv4b[k])) {
                    acc = acc + S_.at(u + du4b[k], v + dv4b[k]);
                    ++n;
                }
            if (n > 0) paused_.push_back({id, acc / static_cast<f32>(n)});
            else queued_[static_cast<usize>(id)] = 0;
        }
        return total_placed;
    }
};

}  // namespace fenix::segment
