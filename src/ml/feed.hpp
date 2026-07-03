// ml/feed.hpp — `fenix train-feed`: the training data plane (torch-free, always built).
// Streams (CT, GT-band[, teacher-prob]) u8 patch triples into a mmap'd shared-memory ring
// that the Python learning loop (tools/train/) consumes zero-copy as numpy tensors — see
// docs/design/training-pipeline.md. Producer threads: deterministic sampler draw →
// gather_box from the CT .fxvol → rasterize the GT band from the .fxsurf → (optional)
// octahedral augmentation applied identically to all channels (exact voxel permutation,
// label-safe; elastic/intensity stay in the Python loop or a later coord-transform pass).
//
// Ring protocol (single file, header + nslots fixed-size slots):
//   slot.state: 0=FREE, 2=WRITING (producer CAS), 1=READY (release-stored after fill).
//   Python: scan for READY, consume, plain-store FREE. u32 transitions on x86/arm64 are
//   atomic at this granularity; the C++ side uses std::atomic_ref with acq/rel.
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"
#include "ml/augment.hpp"
#include "ml/rasterize.hpp"
#include "ml/sampler.hpp"
#include "ml/surface_index.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace fenix::ml {

namespace detail {

struct RingHeader {
    char magic[8];  // "FXRING1\0"
    u32 version;    // 1
    u32 nslots;
    u64 patch;       // P (slot tensors are P^3)
    u64 slot_bytes;  // total bytes per slot incl. slot header
    u32 channels;    // 2 = CT,GT  |  3 = CT,GT,teacher
    u32 reserved;
};
inline constexpr u64 kRingHdr = 4096;
inline constexpr u64 kSlotHdr = 64;
inline constexpr u32 kFree = 0, kReady = 1, kWriting = 2;

struct SlotHeader {
    u32 state;
    u32 mesh;
    u64 draw;
    s64 origin[3];  // patch origin in the CT volume (ZYX)
};

struct FeedPair {
    std::string surf_path, ct_path, teacher_path;  // teacher optional (empty)
    Index3 crop{0, 0, 0};  // CT crop origin in scroll coords (mesh coords are absolute; shifted at load)
    f32 um = 2.4f;         // source voxel size; != 2.4 -> resampled to the canonical 2.4 um grid
};

// pairs file, one entry per line:
//   <fxsurf> <ct> [teacher.fxvol|-] [crop_z crop_y crop_x] [um=<voxel-um>]
// um: the source volume's voxel size. The TRAINING GRID IS CANONICALLY 2.4 um (decided
// 2026-07-02): other resolutions are resampled at feed time — CT/teacher trilinearly, GT
// exactly (surface coords are scaled into canonical space BEFORE rasterization, so labels
// are never interpolated).
// <ct> is either a local .fxvol, or an ON-DEMAND CACHE: '<cache.fxvol>@<zarr-level-root-url>' —
// chunks are pulled from the zarr the first time a patch needs them, appended into the cache
// archive, and served locally forever after (io/cached_volume.hpp). With a cache, mesh coords
// are absolute scroll coords and no crop is needed.
// '-' = no teacher. crop origin: where a CROPPED local CT sits in the scroll volume (0 0 0 = full).
inline Expected<std::vector<FeedPair>> read_pairs(const std::string& path) {
    std::ifstream f(path);
    if (!f) return err(Errc::not_found, "train-feed: cannot open pairs file " + path);
    std::vector<FeedPair> out;
    std::string line;
    while (std::getline(f, line)) {
        const auto h = line.find('#');
        if (h != std::string::npos) line.resize(h);
        std::vector<std::string> tok;
        for (usize p = 0; p < line.size();) {
            while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
            usize e = p;
            while (e < line.size() && line[e] != ' ' && line[e] != '\t') ++e;
            if (e > p) tok.push_back(line.substr(p, e - p));
            p = e;
        }
        if (tok.empty()) continue;
        if (tok.size() < 2) return err(Errc::invalid_argument, "train-feed: bad pairs line: " + line);
        FeedPair fp;
        // um=<v> may appear as the last token of any line form
        if (!tok.empty() && tok.back().rfind("um=", 0) == 0) {
            const std::string t = tok.back().substr(3);
            std::from_chars(t.data(), t.data() + t.size(), fp.um);
            if (!(fp.um > 0)) return err(Errc::invalid_argument, "train-feed: bad um= on line: " + line);
            tok.pop_back();
        }
        fp.surf_path = tok[0];
        fp.ct_path = tok[1];
        if (tok.size() > 2 && tok[2] != "-") fp.teacher_path = tok[2];
        if (tok.size() >= 6) {
            auto pi = [](const std::string& t) {
                s64 v = 0;
                std::from_chars(t.data(), t.data() + t.size(), v);
                return v;
            };
            fp.crop = Index3{pi(tok[3]), pi(tok[4]), pi(tok[5])};
        } else if (tok.size() != 2 && tok.size() != 3) {
            return err(Errc::invalid_argument, "train-feed: bad pairs line (want 2, 3, or 6 tokens): " + line);
        }
        out.push_back(std::move(fp));
    }
    if (out.empty()) return err(Errc::invalid_argument, "train-feed: pairs file has no entries");
    return out;
}

struct Ring {
    int fd = -1;
    u8* base = nullptr;
    u64 total = 0;
    RingHeader* hdr = nullptr;

    static Expected<Ring> create(const std::string& path, u32 nslots, u64 patch, u32 channels) {
        Ring r;
        const u64 tensor = patch * patch * patch;
        const u64 slot = kSlotHdr + tensor * channels;
        r.total = kRingHdr + slot * nslots;
        // ATTACH-or-create: O_TRUNC on an existing ring SIGBUSes a live consumer whose mmap'd
        // pages vanish (measured: a restarted feeder killed the training loop). If a compatible
        // ring already exists, adopt it — reset stale WRITING slots (a dead producer's
        // half-writes) to FREE and leave READY slots for the consumer.
        if (std::filesystem::exists(path)) {
            r.fd = ::open(path.c_str(), O_RDWR, 0644);
            if (r.fd < 0) return err(Errc::io_error, "train-feed: cannot open ring " + path);
            RingHeader h{};
            if (::pread(r.fd, &h, sizeof h, 0) == static_cast<ssize_t>(sizeof h) &&
                std::memcmp(h.magic, "FXRING1\0", 8) == 0 && h.version == 1 && h.nslots == nslots && h.patch == patch &&
                h.slot_bytes == slot && h.channels == channels) {
                void* m = ::mmap(nullptr, r.total, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd, 0);
                if (m == MAP_FAILED) {
                    ::close(r.fd);
                    return err(Errc::io_error, "train-feed: mmap failed for " + path);
                }
                r.base = static_cast<u8*>(m);
                r.hdr = reinterpret_cast<RingHeader*>(r.base);
                for (u32 s2 = 0; s2 < nslots; ++s2) {
                    auto st = std::atomic_ref<u32>(*state_ptr(r, s2));
                    u32 expect = kWriting;
                    st.compare_exchange_strong(expect, kFree);
                }
                return r;
            }
            ::close(r.fd);  // incompatible ring: fall through and rebuild it
        }
        r.fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (r.fd < 0) return err(Errc::io_error, "train-feed: cannot create ring " + path);
        if (::ftruncate(r.fd, static_cast<off_t>(r.total)) != 0) {
            ::close(r.fd);
            return err(Errc::io_error, "train-feed: ftruncate failed for " + path);
        }
        void* m = ::mmap(nullptr, r.total, PROT_READ | PROT_WRITE, MAP_SHARED, r.fd, 0);
        if (m == MAP_FAILED) {
            ::close(r.fd);
            return err(Errc::io_error, "train-feed: mmap failed for " + path);
        }
        r.base = static_cast<u8*>(m);
        std::memset(r.base, 0, kRingHdr);
        r.hdr = reinterpret_cast<RingHeader*>(r.base);
        std::memcpy(r.hdr->magic, "FXRING1\0", 8);
        r.hdr->version = 1;
        r.hdr->nslots = nslots;
        r.hdr->patch = patch;
        r.hdr->slot_bytes = slot;
        r.hdr->channels = channels;
        for (u32 s = 0; s < nslots; ++s) std::atomic_ref<u32>(*state_ptr(r, s)).store(kFree);
        return r;
    }
    static u32* state_ptr(Ring& r, u32 s) { return reinterpret_cast<u32*>(r.base + kRingHdr + s * r.hdr->slot_bytes); }
    [[nodiscard]] u8* slot_data(u32 s) const { return base + kRingHdr + s * hdr->slot_bytes; }
    void close() {
        if (base) ::munmap(base, total);
        if (fd >= 0) ::close(fd);
        base = nullptr;
        fd = -1;
    }
};

}  // namespace detail

// fenix train-feed <pairs.txt> <ring-path> [patch=256] [slots=16] [seed=42] [threads=8]
//                  [octa=1] [thickness=2] [count=0 (0 = run until killed)]
inline Expected<int> run_train_feed(std::span<const std::string_view> args, Context&) {
    using namespace detail;
    if (args.size() < 2)
        return err(Errc::invalid_argument,
                   "usage: train-feed <pairs.txt> <ring> [patch=] [slots=] [seed=] [threads=] [octa=] "
                   "[aug=1 (0=off 1=octa 2=full policy)] [thickness=] [count=] [cache_mb=4096] [locality=16]");
    s64 patch = 256, slots = 16, threads = 8, count = 0, cache_mb = 4096, locality = 16, prefetch = 256,
        disk_mb = 32768,  // per-volume DISK cap for on-demand caches (reset-on-full; 0 = unbounded)
        echo = 1;         // data echoing: emissions per draw, each independently augmented (2-4 when
                          // the feed is network/gather-bound; correlated samples are the known cost)
    u64 seed = 42;
    int octa = -1, aug_mode = 1;  // octa= is the legacy alias for aug=
    // cache_q=32: training tolerates a soft CT cache (forrest 2026-07-03 — the compression aug
    // trains robustness to exactly this artifact class) and it's ~4x less disk than q=8.
    f32 thickness = 2.0f, so3 = 0.0f, cache_q = 32.0f;
    for (usize i = 2; i < args.size(); ++i) {
        const auto kv = args[i];
        auto num = [&](std::string_view key, auto& dst) -> bool {
            if (!kv.starts_with(key) || kv.size() <= key.size() || kv[key.size()] != '=') return false;
            const auto t = kv.substr(key.size() + 1);
            std::from_chars(t.data(), t.data() + t.size(), dst);
            return true;
        };
        if (num("patch", patch) || num("slots", slots) || num("seed", seed) || num("threads", threads) ||
            num("octa", octa) || num("aug", aug_mode) || num("thickness", thickness) || num("count", count) ||
            num("cache_mb", cache_mb) || num("locality", locality) || num("so3", so3) || num("cache_q", cache_q) ||
            num("prefetch", prefetch) || num("disk_mb", disk_mb) || num("echo", echo))
            continue;
        return err(Errc::invalid_argument, "train-feed: unknown arg '" + std::string(kv) + "'");
    }
    if (octa >= 0) aug_mode = octa;  // legacy alias: octa=0/1 maps to aug=0/1
    if (aug_mode < 0 || aug_mode > 2) return err(Errc::invalid_argument, "train-feed: aug wants 0, 1 or 2");
    echo = std::clamp<s64>(echo, 1, 16);

    auto pairs = read_pairs(std::string(args[0]));
    if (!pairs) return std::unexpected(pairs.error());

    // Concurrent CachedVolume fills each run their own parallel zarr fetch: N workers x the
    // reader's default 24 fetch threads self-congests one S3 endpoint until per-transfer speed
    // trips the stall watchdog (measured: patch=256, 8 workers -> ~200 transfers -> curl
    // timeout). Cap the PER-FILL parallelism; the fills themselves already parallelize.
    ::setenv("FENIX_ZARR_FETCH_THREADS", "6", /*overwrite=*/0);

    // Load meshes + open archives, GROUPED BY CT VOLUME: multiple segments routinely live in
    // the same training chunk (81 PHercParis4 segments share one scroll), so the GT band for a
    // patch must be the union of EVERY mesh on that volume — labeling only the sampled mesh
    // would mark the other segments' true sheets as background. Meshes stay resident; archives
    // decode tiles on demand through their block cache.
    struct Entry {
        std::vector<Surface> surfs;  // every mesh on this CT volume (crop-shifted, CANONICAL coords)
        std::vector<const Surface*> surf_ptrs;
        VolumeSurfaceIndex index;                   // R-tree: which meshes touch a patch, and where
        std::optional<codec::VolumeArchive> ct;     // local .fxvol ...
        std::optional<io::CachedVolume> ct_cached;  // ... or on-demand zarr-backed cache
        std::optional<codec::VolumeArchive> teacher;
        Extent3 dims{};    // CANONICAL (2.4 um) dims
        f32 scale = 1.0f;  // source voxels per canonical voxel (= um / 2.4)

        Expected<void> gather_src_u8_(codec::VolumeArchive* teach, s64 oz, s64 oy, s64 ox, s64 n, u8* out) {
            if (teach) return teach->gather_box_u8(0, oz, oy, ox, n, n, n, out);
            return ct_cached ? ct_cached->gather_box_u8(oz, oy, ox, n, n, n, out)
                             : ct->gather_box_u8(0, oz, oy, ox, n, n, n, out);
        }

        Expected<void> gather_src_(codec::VolumeArchive* teach, s64 oz, s64 oy, s64 ox, s64 D, s64 H, s64 W, f32* out) {
            if (teach) return teach->gather_box_f32(0, oz, oy, ox, D, H, W, out);
            return ct_cached ? ct_cached->gather_box_f32(oz, oy, ox, D, H, W, out)
                             : ct->gather_box_f32(0, oz, oy, ox, D, H, W, out);
        }

        // Gather an n^3 CANONICAL patch at `org` (canonical coords): direct at scale 1, else a
        // scaled source box trilinearly resampled onto the canonical grid (upsample for coarser
        // sources like 7.91 um; plain trilinear also handles the mild 1.129->2.4 downsample —
        // box-filtering is a TODO if aliasing shows in training).
        Expected<void> gather_canonical(bool teach, Index3 org, s64 n, f32* out, std::vector<f32>& scratch) {
            codec::VolumeArchive* t = teach ? &*teacher : nullptr;
            if (scale == 1.0f) return gather_src_(t, org.z, org.y, org.x, n, n, n, out);
            const f64 sc = static_cast<f64>(scale);
            const s64 s0z = static_cast<s64>(std::floor(static_cast<f64>(org.z) * sc)) - 1;
            const s64 s0y = static_cast<s64>(std::floor(static_cast<f64>(org.y) * sc)) - 1;
            const s64 s0x = static_cast<s64>(std::floor(static_cast<f64>(org.x) * sc)) - 1;
            const s64 sn = static_cast<s64>(std::ceil(static_cast<f64>(n) * sc)) + 3;
            scratch.resize(static_cast<usize>(sn * sn * sn));
            if (auto r = gather_src_(t, s0z, s0y, s0x, sn, sn, sn, scratch.data()); !r) return r;
            auto sample = [&](f64 z, f64 y, f64 x) {
                const s64 iz = std::clamp<s64>(static_cast<s64>(z), 0, sn - 2);
                const s64 iy = std::clamp<s64>(static_cast<s64>(y), 0, sn - 2);
                const s64 ix = std::clamp<s64>(static_cast<s64>(x), 0, sn - 2);
                const f32 fz = static_cast<f32>(z - static_cast<f64>(iz));
                const f32 fy = static_cast<f32>(y - static_cast<f64>(iy));
                const f32 fx = static_cast<f32>(x - static_cast<f64>(ix));
                auto at = [&](s64 a, s64 b, s64 c) { return scratch[static_cast<usize>((a * sn + b) * sn + c)]; };
                const f32 c00 = at(iz, iy, ix) * (1 - fx) + at(iz, iy, ix + 1) * fx;
                const f32 c01 = at(iz, iy + 1, ix) * (1 - fx) + at(iz, iy + 1, ix + 1) * fx;
                const f32 c10 = at(iz + 1, iy, ix) * (1 - fx) + at(iz + 1, iy, ix + 1) * fx;
                const f32 c11 = at(iz + 1, iy + 1, ix) * (1 - fx) + at(iz + 1, iy + 1, ix + 1) * fx;
                return (c00 * (1 - fy) + c01 * fy) * (1 - fz) + (c10 * (1 - fy) + c11 * fy) * fz;
            };
            for (s64 z = 0; z < n; ++z)
                for (s64 y = 0; y < n; ++y)
                    for (s64 x = 0; x < n; ++x) {
                        const f64 gz = (static_cast<f64>(org.z + z) + 0.5) * sc - 0.5 - static_cast<f64>(s0z);
                        const f64 gy = (static_cast<f64>(org.y + y) + 0.5) * sc - 0.5 - static_cast<f64>(s0y);
                        const f64 gx = (static_cast<f64>(org.x + x) + 0.5) * sc - 0.5 - static_cast<f64>(s0x);
                        out[(z * n + y) * n + x] = sample(gz, gy, gx);
                    }
            return {};
        }
    };
    std::vector<Entry> entries;
    std::vector<std::string> entry_key;
    bool any_teacher = false;
    for (const auto& p : *pairs) {
        auto s = io::read_fxsurf(p.surf_path);
        if (!s) return std::unexpected(s.error());
        if (p.crop.z != 0 || p.crop.y != 0 || p.crop.x != 0) {  // absolute mesh coords -> crop space
            const Vec3f off{static_cast<f32>(p.crop.z), static_cast<f32>(p.crop.y), static_cast<f32>(p.crop.x)};
            for (auto& c : s->coord) c = c - off;
        }
        usize ei = 0;
        const std::string key = p.ct_path + "@um=" + std::to_string(p.um);
        for (; ei < entry_key.size(); ++ei)
            if (entry_key[ei] == key) break;
        if (ei == entry_key.size()) {
            Entry e;
            const auto at = p.ct_path.find('@');
            if (at != std::string::npos) {  // '<cache.fxvol>@<zarr-level-root>' = on-demand
                auto cv = io::CachedVolume::open(p.ct_path.substr(0, at), p.ct_path.substr(at + 1), cache_q);
                if (!cv) return std::unexpected(cv.error());
                e.dims = cv->dims();
                e.ct_cached = std::move(*cv);
            } else {
                auto a = codec::VolumeArchive::open(p.ct_path);
                if (!a) return std::unexpected(a.error());
                e.dims = a->dims();
                e.ct = std::move(*a);
            }
            if (!p.teacher_path.empty()) {
                auto t = codec::VolumeArchive::open(p.teacher_path);
                if (!t) return std::unexpected(t.error());
                e.teacher = std::move(*t);
                any_teacher = true;
            }
            e.scale = p.um / 2.4f;
            if (e.scale != 1.0f)
                e.dims = Extent3{static_cast<s64>(static_cast<f64>(e.dims.z) / e.scale),
                                 static_cast<s64>(static_cast<f64>(e.dims.y) / e.scale),
                                 static_cast<s64>(static_cast<f64>(e.dims.x) / e.scale)};
            entries.push_back(std::move(e));
            entry_key.push_back(key);
        }
        if (entries[ei].scale != 1.0f) {  // mesh coords -> canonical 2.4 um space
            const f32 inv = 1.0f / entries[ei].scale;
            for (auto& c : s->coord) c = c * inv;
            s->scale_u *= inv;  // grid step is in voxels; keep it consistent in canonical units
            s->scale_v *= inv;
        }
        entries[ei].surfs.push_back(std::move(*s));
    }
    const u32 channels = any_teacher ? 3u : 2u;

    // Decoded-block cache budget PER VOLUME: the 256 MiB archive default thrashes under training
    // (measured: warm gathers degrade 81ms -> 336ms as eviction sets in; a 400-draw window's
    // working set is ~3 GiB). Sized here, before the worker threads exist (reserve_cache contract).
    for (auto& e : entries) {
        if (e.ct_cached) {
            e.ct_cached->reserve_cache(static_cast<u64>(cache_mb) << 20);
            e.ct_cached->disk_budget(static_cast<u64>(disk_mb) << 20);
        } else {
            e.ct->reserve_cache(static_cast<u64>(cache_mb) << 20);
        }
        if (e.teacher) e.teacher->reserve_cache((static_cast<u64>(cache_mb) << 20) / 4);
    }

    // sampler runs over ALL meshes; mesh index -> owning entry
    std::vector<const Surface*> meshes;
    std::vector<usize> mesh_entry;
    for (usize ei = 0; ei < entries.size(); ++ei) {
        for (auto& sf : entries[ei].surfs) {
            meshes.push_back(&sf);
            mesh_entry.push_back(ei);
        }
        for (auto& sf : entries[ei].surfs) entries[ei].surf_ptrs.push_back(&sf);
        entries[ei].index = VolumeSurfaceIndex(entries[ei].surf_ptrs);
    }
    PatchSampler sampler(meshes,
                         seed,
                         static_cast<f32>(patch) / 4.0f,
                         locality <= 1 ? 0u : static_cast<u32>(locality),
                         static_cast<f32>(patch) * 1.5f);
    if (sampler.total_weight() <= 0) return err(Errc::invalid_argument, "train-feed: corpus has no valid cells");

    auto ring = Ring::create(std::string(args[1]), static_cast<u32>(slots), static_cast<u64>(patch), channels);
    if (!ring) return std::unexpected(ring.error());
    log(LogLevel::info,
        "train-feed: {} pairs, {} slots x {} B, patch={} channels={} -> {}",
        entries.size(),
        slots,
        ring->hdr->slot_bytes,
        patch,
        channels,
        args[1]);

    const u64 tensor = static_cast<u64>(patch) * static_cast<u64>(patch) * static_cast<u64>(patch);
    const Extent3 pext{patch, patch, patch};
    std::atomic<u64> next{0};
    std::atomic<bool> failed{false};
    // per-stage wall-time accounting (ms, summed across workers) — printed at exit so every
    // run reports where feed time went (gather = CT fetch+decode, raster = GT stamp,
    // aug = octahedral, slotwait = ring full / consumer slow)
    std::atomic<u64> t_gather{0}, t_raster{0}, t_aug{0}, t_slotwait{0}, n_done{0};
    std::string fail_msg;
    std::mutex fail_mu;

    auto now_ms = [] {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
    };
    auto worker = [&] {
        // Workers ARE the parallelism: everything they call (augment's parallel_for_z, the
        // rasterizer, archive decode) runs serial on this thread — 12+ workers each spawning
        // inner OpenMP teams oversubscribes the box (core/parallel.hpp measured ~10x).
        SerialRegion serial;
        std::vector<f32> fbuf(static_cast<usize>(tensor));
        std::vector<f32> scratch;
        // Staging: gather+rasterize ONCE per draw here, then emit `echo` independently-augmented
        // copies (data echoing — amortizes network/gather/raster over K emissions; each copy gets
        // its own aug seed so the GPU sees K distinct views, not duplicates). Also means no ring
        // slot sits in WRITING through a slow S3 gather.
        std::vector<u8> stage_ct(static_cast<usize>(tensor)), stage_gt(static_cast<usize>(tensor)),
            stage_te(channels == 3 ? static_cast<usize>(tensor) : 0);
        while (!failed.load(std::memory_order_relaxed)) {
            const u64 i = next.fetch_add(1);
            if (count > 0 && i * static_cast<u64>(echo) >= static_cast<u64>(count)) return;
            const u64 n_emit =
                count > 0 ? std::min<u64>(static_cast<u64>(echo), static_cast<u64>(count) - i * static_cast<u64>(echo))
                          : static_cast<u64>(echo);
            const PatchDraw d = sampler.draw(i);
            Entry& e = entries[mesh_entry[static_cast<usize>(d.mesh)]];
            const Index3 org = PatchSampler::patch_origin(d.center, pext, e.dims);
            u8* ct_out = stage_ct.data();
            u8* gt_out = stage_gt.data();
            u8* te_out = channels == 3 ? stage_te.data() : nullptr;

            auto fail = [&](const Error& er) {
                std::lock_guard<std::mutex> lk(fail_mu);
                if (!failed.exchange(true)) fail_msg = er.message;
            };
            // One retry on a transient fetch failure (S3 stall/timeout) before declaring the
            // feeder dead — a multi-hour training run shouldn't die to a single flaky transfer.
            const auto tg0 = now_ms();
            Expected<void> g{};
            if (e.scale == 1.0f && e.dims.z >= patch && e.dims.y >= patch &&
                e.dims.x >= patch) {  // u8-native fast path
                g = e.gather_src_u8_(nullptr, org.z, org.y, org.x, patch, ct_out);
                if (!g) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    g = e.gather_src_u8_(nullptr, org.z, org.y, org.x, patch, ct_out);
                }
            } else {
                g = e.gather_canonical(false, org, patch, fbuf.data(), scratch);
                if (!g) {
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                    g = e.gather_canonical(false, org, patch, fbuf.data(), scratch);
                }
                if (g)
                    for (u64 k = 0; k < tensor; ++k)
                        ct_out[k] = static_cast<u8>(std::clamp(fbuf[k], 0.0f, 255.0f) + 0.5f);
            }
            if (!g) return fail(g.error());
            t_gather += static_cast<u64>(now_ms() - tg0);
            const auto tr0 = now_ms();

            // union band over EVERY mesh on this volume + trusted-background shell (tri-state:
            // 255 sheet / 128 background / 0 unlabeled-ignore)
            Volume<u8> band = rasterize_band_multi(
                e.surf_ptrs, org, pext, {.thickness = thickness, .shell = std::min(thickness * 4, 16.0f)}, &e.index);
            std::memcpy(gt_out, band.flat().data(), tensor);
            // INTENSITY-GATE the shell (finding 2026-07-03-mesh-volume-misalignment): at 2.4 um the
            // wraps sit ~15-40 vox apart, so a geometric shell at ±16 lands ON the neighboring wrap
            // (traced or not) — labeling bright sheet material "trusted background" makes the classes
            // statistically identical and training collapses to a constant (measured: sheet-vs-shell
            // CT delta +1.3 while sheet-vs-everything was +12.5). Background must also be DARK:
            // shell voxels brighter than the patch's sheet-level stay unlabeled-ignore.
            {
                f64 s_sum = 0;
                u64 s_n = 0;
                for (u64 kk = 0; kk < tensor; ++kk)
                    if (gt_out[kk] == kLabelSheet) {
                        s_sum += ct_out[kk];
                        ++s_n;
                    }
                if (s_n > 256) {
                    // gate: dark = below halfway between the patch mean and the sheet mean
                    f64 c_sum = 0;
                    for (u64 kk = 0; kk < tensor; ++kk) c_sum += ct_out[kk];
                    const f64 sheet_mean = s_sum / static_cast<f64>(s_n);
                    const f64 vol_mean = c_sum / static_cast<f64>(tensor);
                    const u8 gate = static_cast<u8>(std::clamp((sheet_mean + vol_mean) * 0.5, 1.0, 254.0));
                    for (u64 kk = 0; kk < tensor; ++kk)
                        if (gt_out[kk] == kLabelBackground && ct_out[kk] >= gate) gt_out[kk] = kLabelUnknown;
                }
            }
            t_raster += static_cast<u64>(now_ms() - tr0);

            if (te_out) {
                if (e.scale == 1.0f) {
                    if (auto r = e.gather_src_u8_(&*e.teacher, org.z, org.y, org.x, patch, te_out); !r)
                        return fail(r.error());
                } else {
                    if (auto r = e.gather_canonical(true, org, patch, fbuf.data(), scratch); !r) return fail(r.error());
                    for (u64 k = 0; k < tensor; ++k)
                        te_out[k] = static_cast<u8>(std::clamp(fbuf[k], 0.0f, 255.0f) + 0.5f);
                }
            }

            // Emit `n_emit` independently-augmented copies of the staged draw (data echoing).
            // Aug seeds derive from (seed, i, k) — distinct per emission, deterministic overall.
            // octa (aug=1, default): the SAME exact voxel permutation on every channel — flips/
            // 90° rotations are label-safe. aug=2: the FULL policy chain (ml/augment.hpp —
            // octahedral + rotate_z + [so3] + scale + elastic + ct_degrade + lowres + compression
            // + intensity); geometric ops move CT/GT/teacher together (GT nearest, teacher
            // trilinear), image-only corruptions never touch GT/teacher.
            for (u64 em = 0; em < n_emit && !failed.load(std::memory_order_relaxed); ++em) {
                const u64 eid = i * static_cast<u64>(echo) + em;
                // claim a free slot (spin over the ring; the consumer frees them)
                const auto tw0 = now_ms();
                u32 s = static_cast<u32>(eid % static_cast<u64>(slots));
                for (;;) {
                    u32 expect = kFree;
                    if (std::atomic_ref<u32>(*Ring::state_ptr(*ring, s))
                            .compare_exchange_strong(expect, kWriting, std::memory_order_acquire))
                        break;
                    s = (s + 1) % static_cast<u32>(slots);
                    if (s == eid % static_cast<u64>(slots)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (failed.load(std::memory_order_relaxed)) return;
                }
                t_slotwait += static_cast<u64>(now_ms() - tw0);
                u8* base = ring->slot_data(s);
                auto* sh = reinterpret_cast<SlotHeader*>(base);
                sh->mesh = static_cast<u32>(d.mesh);
                sh->draw = eid;
                sh->origin[0] = org.z;
                sh->origin[1] = org.y;
                sh->origin[2] = org.x;
                u8* ct_s = base + kSlotHdr;
                u8* gt_s = ct_s + tensor;
                u8* te_s = channels == 3 ? gt_s + tensor : nullptr;
                std::memcpy(ct_s, ct_out, tensor);
                std::memcpy(gt_s, gt_out, tensor);
                if (te_s) std::memcpy(te_s, te_out, tensor);

                const u64 aseed = hash_value(std::array<u64, 3>{seed ^ 0xa5a5a5a5ull, i, em});
                if (aug_mode == 2) {
                    const auto ta0 = now_ms();
                    aug::Sample smp;
                    smp.image = Volume<f32>(pext);
                    for (u64 k = 0; k < tensor; ++k) smp.image.flat()[k] = static_cast<f32>(ct_s[k]);
                    smp.label = Volume<u8>::zeros(pext);
                    std::memcpy(smp.label.flat().data(), gt_s, tensor);
                    if (te_s) {
                        smp.teacher = Volume<f32>(pext);
                        for (u64 k = 0; k < tensor; ++k) smp.teacher.flat()[k] = static_cast<f32>(te_s[k]);
                    }
                    aug::Policy pol;
                    pol.p_so3 = so3;  // so3=<prob> feeder knob — the full-SO(3) experiment (default 0)
                    aug::augment(smp, aseed, pol);
                    for (u64 k = 0; k < tensor; ++k)
                        ct_s[k] = static_cast<u8>(std::clamp(smp.image.flat()[k], 0.0f, 255.0f) + 0.5f);
                    std::memcpy(gt_s, smp.label.flat().data(), tensor);
                    if (te_s)
                        for (u64 k = 0; k < tensor; ++k)
                            te_s[k] = static_cast<u8>(std::clamp(smp.teacher.flat()[k], 0.0f, 255.0f) + 0.5f);
                    t_aug += static_cast<u64>(now_ms() - ta0);
                } else if (aug_mode == 1) {
                    const auto ta0 = now_ms();
                    const int member = static_cast<int>(aseed % 48);
                    if (member != 0) {
                        aug::octahedral_u8_inplace(ct_s, patch, member);
                        aug::octahedral_u8_inplace(gt_s, patch, member);
                        if (te_s) aug::octahedral_u8_inplace(te_s, patch, member);
                    }
                    t_aug += static_cast<u64>(now_ms() - ta0);
                }
                std::atomic_ref<u32>(*Ring::state_ptr(*ring, s)).store(kReady, std::memory_order_release);
                ++n_done;
            }
        }
    };

    // PREFETCH-AHEAD: the sampler is deterministic, so future draws' locations are known NOW.
    // Prefetchers walk `prefetch` draws ahead of the workers and ensure() the chunks (S3 fetch +
    // cache write) before any worker asks — no draw blocks on the network except when prefetch
    // falls behind raw bandwidth. ensure()'s in-flight claim set dedups against worker fetches.
    std::atomic<u64> pnext{0};
    auto prefetcher = [&] {
        SerialRegion serial;  // fetch fan-out is parallel_for_io (std::threads) — unaffected;
                              // this only stops OMP decode teams under the prefetch threads
        while (!failed.load(std::memory_order_relaxed)) {
            const u64 j = pnext.fetch_add(1);  // draw index; count is in EMITTED patches
            if (count > 0 && j * static_cast<u64>(echo) >= static_cast<u64>(count)) return;
            while (j > next.load(std::memory_order_relaxed) + static_cast<u64>(prefetch)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                if (failed.load(std::memory_order_relaxed)) return;
            }
            const PatchDraw d = sampler.draw(j);
            if (d.mesh < 0) continue;
            Entry& e = entries[mesh_entry[static_cast<usize>(d.mesh)]];
            if (!e.ct_cached) continue;  // local archives have nothing to prefetch
            const Index3 org = PatchSampler::patch_origin(d.center, pext, e.dims);
            Index3 so = org;
            Extent3 se = pext;
            if (e.scale != 1.0f) {  // mirror gather_canonical's source-box math
                const f64 sc = static_cast<f64>(e.scale);
                so = Index3{static_cast<s64>(std::floor(static_cast<f64>(org.z) * sc)) - 1,
                            static_cast<s64>(std::floor(static_cast<f64>(org.y) * sc)) - 1,
                            static_cast<s64>(std::floor(static_cast<f64>(org.x) * sc)) - 1};
                const s64 sn = static_cast<s64>(std::ceil(static_cast<f64>(patch) * sc)) + 3;
                se = Extent3{sn, sn, sn};
            }
            (void)e.ct_cached->ensure(so, se);  // failure is fine: the worker path retries properly
        }
    };

    std::vector<std::thread> pool;
    const s64 n_prefetch = prefetch > 0 ? std::clamp<s64>(threads / 4, 1, 4) : 0;
    for (s64 t = 0; t < n_prefetch; ++t) pool.emplace_back(prefetcher);
    for (s64 t = 0; t < std::max<s64>(1, threads); ++t) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    log(LogLevel::info,
        "train-feed: {} draws done | worker-time gather={}s raster={}s aug={}s slotwait={}s",
        n_done.load(),
        t_gather.load() / 1000,
        t_raster.load() / 1000,
        t_aug.load() / 1000,
        t_slotwait.load() / 1000);
    ring->close();
    if (failed.load()) return err(Errc::internal, "train-feed: " + fail_msg);
    log(LogLevel::info, "train-feed: done ({} draws)", count);
    return 0;
}

}  // namespace fenix::ml

namespace {
[[maybe_unused]] const int fenix_stage_train_feed =
    ::fenix::register_stage(::fenix::Stage{"train-feed",
                                           "stream training patch triples (CT, GT band[, teacher]) into a shm ring",
                                           ::fenix::ml::run_train_feed});
}  // namespace
