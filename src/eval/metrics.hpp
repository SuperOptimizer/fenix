// eval/metrics.hpp — segmentation quality metrics. Dice/IoU (overlap) + VOI (Variation of
// Information, the instance-segmentation metric where a cross-wrap merge is catastrophic).
// More (NSD, surface-Dice, TopoScore via topo/, winding consistency) per eval/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include <cmath>
#include <unordered_map>

namespace fenix::eval {

// Binary Dice = 2|A∩B| / (|A|+|B|). Inputs are foreground masks (!=0).
inline f64 dice(VolumeView<const u8> a, VolumeView<const u8> b) {
    s64 inter = 0, na = 0, nb = 0;
    const s64 n = a.size();
    for (s64 i = 0; i < n; ++i) {
        const bool fa = a.flat()[static_cast<usize>(i)] != 0;
        const bool fb = b.flat()[static_cast<usize>(i)] != 0;
        inter += (fa && fb) ? 1 : 0;
        na += fa ? 1 : 0;
        nb += fb ? 1 : 0;
    }
    if (na + nb == 0) return 1.0;  // both empty -> identical
    return 2.0 * static_cast<f64>(inter) / static_cast<f64>(na + nb);
}

inline f64 iou(VolumeView<const u8> a, VolumeView<const u8> b) {
    s64 inter = 0, uni = 0;
    const s64 n = a.size();
    for (s64 i = 0; i < n; ++i) {
        const bool fa = a.flat()[static_cast<usize>(i)] != 0;
        const bool fb = b.flat()[static_cast<usize>(i)] != 0;
        inter += (fa && fb) ? 1 : 0;
        uni += (fa || fb) ? 1 : 0;
    }
    return uni == 0 ? 1.0 : static_cast<f64>(inter) / static_cast<f64>(uni);
}

// Variation of Information between two integer label volumes (0 = ignore/background).
// VOI = H(seg|gt) + H(gt|seg); 0 == identical partition. Lower is better.
struct Voi {
    f64 split = 0;  // H(seg|gt) — over-segmentation
    f64 merge = 0;  // H(gt|seg) — under-segmentation (a cross-wrap merge: catastrophic)
    [[nodiscard]] f64 total() const { return split + merge; }
};

inline Voi voi(VolumeView<const s32> seg, VolumeView<const s32> gt) {
    std::unordered_map<s64, s64> ns, ng;       // marginal counts
    std::unordered_map<s64, s64> nsg;          // joint counts, key = seg<<32 | gt
    s64 total = 0;
    const s64 n = seg.size();
    for (s64 i = 0; i < n; ++i) {
        const s32 s = seg.flat()[static_cast<usize>(i)];
        const s32 g = gt.flat()[static_cast<usize>(i)];
        if (s == 0 || g == 0) continue;  // ignore background in both
        ++ns[s];
        ++ng[g];
        ++nsg[(static_cast<s64>(s) << 32) | static_cast<u32>(g)];
        ++total;
    }
    if (total == 0) return {};
    const f64 N = static_cast<f64>(total);
    Voi r;
    for (auto& [key, c] : nsg) {
        const s32 s = static_cast<s32>(key >> 32);
        const s32 g = static_cast<s32>(key & 0xffffffff);
        const f64 pjoint = static_cast<f64>(c) / N;
        const f64 ps = static_cast<f64>(ns[s]) / N;
        const f64 pg = static_cast<f64>(ng[g]) / N;
        // H(seg|gt) = sum p(s,g) log p(g)/p(s,g) (conditions on gt -> over-segmentation/split);
        // H(gt|seg) = sum p(s,g) log p(s)/p(s,g) (conditions on seg -> under-segmentation/merge).
        r.split += pjoint * std::log(pg / pjoint);   // H(seg|gt)
        r.merge += pjoint * std::log(ps / pjoint);   // H(gt|seg)
    }
    return r;
}

}  // namespace fenix::eval
