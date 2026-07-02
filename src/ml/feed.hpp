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
};

// pairs file, one entry per line:
//   <fxsurf> <ct> [teacher.fxvol|-] [crop_z crop_y crop_x]
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
                   "[thickness=] [count=] [cache_mb=4096]");
    s64 patch = 256, slots = 16, threads = 8, count = 0, cache_mb = 4096;
    u64 seed = 42;
    int octa = 1;
    f32 thickness = 2.0f;
    for (usize i = 2; i < args.size(); ++i) {
        const auto kv = args[i];
        auto num = [&](std::string_view key, auto& dst) -> bool {
            if (!kv.starts_with(key) || kv.size() <= key.size() || kv[key.size()] != '=') return false;
            const auto t = kv.substr(key.size() + 1);
            std::from_chars(t.data(), t.data() + t.size(), dst);
            return true;
        };
        if (num("patch", patch) || num("slots", slots) || num("seed", seed) || num("threads", threads) ||
            num("octa", octa) || num("thickness", thickness) || num("count", count) || num("cache_mb", cache_mb))
            continue;
        return err(Errc::invalid_argument, "train-feed: unknown arg '" + std::string(kv) + "'");
    }

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
        std::vector<Surface> surfs;  // every mesh registered on this CT volume (crop-shifted)
        std::vector<const Surface*> surf_ptrs;
        VolumeSurfaceIndex index;                   // R-tree: which meshes touch a patch, and where
        std::optional<codec::VolumeArchive> ct;     // local .fxvol ...
        std::optional<io::CachedVolume> ct_cached;  // ... or on-demand zarr-backed cache
        std::optional<codec::VolumeArchive> teacher;
        Extent3 dims{};

        Expected<void> gather_ct(s64 oz, s64 oy, s64 ox, s64 n, f32* out) {
            return ct_cached ? ct_cached->gather_box_f32(oz, oy, ox, n, n, n, out)
                             : ct->gather_box_f32(0, oz, oy, ox, n, n, n, out);
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
        for (; ei < entry_key.size(); ++ei)
            if (entry_key[ei] == p.ct_path) break;
        if (ei == entry_key.size()) {
            Entry e;
            const auto at = p.ct_path.find('@');
            if (at != std::string::npos) {  // '<cache.fxvol>@<zarr-level-root>' = on-demand
                auto cv = io::CachedVolume::open(p.ct_path.substr(0, at), p.ct_path.substr(at + 1));
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
            entries.push_back(std::move(e));
            entry_key.push_back(p.ct_path);
        }
        entries[ei].surfs.push_back(std::move(*s));
    }
    const u32 channels = any_teacher ? 3u : 2u;

    // Decoded-block cache budget PER VOLUME: the 256 MiB archive default thrashes under training
    // (measured: warm gathers degrade 81ms -> 336ms as eviction sets in; a 400-draw window's
    // working set is ~3 GiB). Sized here, before the worker threads exist (reserve_cache contract).
    for (auto& e : entries) {
        auto& arch = e.ct_cached ? e.ct_cached->archive() : *e.ct;
        arch.reserve_cache(static_cast<u64>(cache_mb) << 20);
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
    PatchSampler sampler(meshes, seed, static_cast<f32>(patch) / 4.0f);
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
        std::vector<f32> fbuf(static_cast<usize>(tensor));
        while (!failed.load(std::memory_order_relaxed)) {
            const u64 i = next.fetch_add(1);
            if (count > 0 && i >= static_cast<u64>(count)) return;
            const PatchDraw d = sampler.draw(i);
            Entry& e = entries[mesh_entry[static_cast<usize>(d.mesh)]];
            const Index3 org = PatchSampler::patch_origin(d.center, pext, e.dims);

            // claim a free slot (spin over the ring; the consumer frees them)
            const auto tw0 = now_ms();
            u32 s = static_cast<u32>(i % static_cast<u64>(slots));
            for (;;) {
                u32 expect = kFree;
                if (std::atomic_ref<u32>(*Ring::state_ptr(*ring, s))
                        .compare_exchange_strong(expect, kWriting, std::memory_order_acquire))
                    break;
                s = (s + 1) % static_cast<u32>(slots);
                if (s == i % static_cast<u64>(slots)) std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (failed.load(std::memory_order_relaxed)) return;
            }
            t_slotwait += static_cast<u64>(now_ms() - tw0);
            u8* base = ring->slot_data(s);
            auto* sh = reinterpret_cast<SlotHeader*>(base);
            sh->mesh = static_cast<u32>(d.mesh);
            sh->draw = i;
            sh->origin[0] = org.z;
            sh->origin[1] = org.y;
            sh->origin[2] = org.x;
            u8* ct_out = base + kSlotHdr;
            u8* gt_out = ct_out + tensor;
            u8* te_out = channels == 3 ? gt_out + tensor : nullptr;

            auto fail = [&](const Error& er) {
                std::lock_guard<std::mutex> lk(fail_mu);
                if (!failed.exchange(true)) fail_msg = er.message;
                std::atomic_ref<u32>(*Ring::state_ptr(*ring, s)).store(kFree, std::memory_order_release);
            };
            // One retry on a transient fetch failure (S3 stall/timeout) before declaring the
            // feeder dead — a multi-hour training run shouldn't die to a single flaky transfer.
            const auto tg0 = now_ms();
            auto g = e.gather_ct(org.z, org.y, org.x, patch, fbuf.data());
            if (!g) {
                std::this_thread::sleep_for(std::chrono::seconds(2));
                g = e.gather_ct(org.z, org.y, org.x, patch, fbuf.data());
            }
            if (!g) return fail(g.error());
            for (u64 k = 0; k < tensor; ++k) ct_out[k] = static_cast<u8>(std::clamp(fbuf[k], 0.0f, 255.0f) + 0.5f);
            t_gather += static_cast<u64>(now_ms() - tg0);
            const auto tr0 = now_ms();

            // union band over EVERY mesh on this volume + trusted-background shell (tri-state:
            // 255 sheet / 128 background / 0 unlabeled-ignore)
            Volume<u8> band = rasterize_band_multi(
                e.surf_ptrs, org, pext, {.thickness = thickness, .shell = std::min(thickness * 4, 16.0f)}, &e.index);
            std::memcpy(gt_out, band.flat().data(), tensor);
            t_raster += static_cast<u64>(now_ms() - tr0);

            if (te_out) {
                if (auto r = e.teacher->gather_box_f32(0, org.z, org.y, org.x, patch, patch, patch, fbuf.data()); !r)
                    return fail(r.error());
                for (u64 k = 0; k < tensor; ++k) te_out[k] = static_cast<u8>(std::clamp(fbuf[k], 0.0f, 255.0f) + 0.5f);
            }

            // Octahedral augmentation: the SAME exact voxel permutation on every channel — flips/
            // 90° rotations are label-safe (no interpolation). Member chosen deterministically.
            if (octa) {
                const auto ta0 = now_ms();
                const int member = static_cast<int>(hash_value(std::array<u64, 2>{seed ^ 0xa5a5a5a5ull, i}) % 48);
                if (member != 0) {
                    aug::octahedral_u8_inplace(ct_out, patch, member);
                    aug::octahedral_u8_inplace(gt_out, patch, member);
                    if (te_out) aug::octahedral_u8_inplace(te_out, patch, member);
                }
                t_aug += static_cast<u64>(now_ms() - ta0);
            }
            std::atomic_ref<u32>(*Ring::state_ptr(*ring, s)).store(kReady, std::memory_order_release);
            ++n_done;
        }
    };

    std::vector<std::thread> pool;
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
