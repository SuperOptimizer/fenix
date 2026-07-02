// io/io.hpp — IO stage: registers the `ingest` subcommand (NRRD -> .fxvol). See io/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "codec/archive.hpp"
#include "io/jpeg.hpp"
#include "io/nrrd.hpp"
#include "io/slice.hpp"
#include "io/zarr.hpp"

#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fenix::io {

inline s64 parse_i(std::string_view s, s64 def) {
    s64 v = def;
    std::from_chars(s.data(), s.data() + s.size(), v);
    return v;
}

// `fenix ingest <in.nrrd> <out.fxvol> [q]` — read a NRRD and transcode to .fxvol (DCT tile codec).
inline Expected<int> ingest(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix ingest <in.nrrd> <out.fxvol> [q=2]");
        return err(Errc::invalid_argument, "missing args");
    }
    auto parse_f = [](std::string_view s, f32 def) {
        f32 v = def;
        std::from_chars(s.data(), s.data() + s.size(), v);
        return v;
    };

    auto vol = read_nrrd(std::string(args[0]));
    if (!vol) return std::unexpected(vol.error());

    codec::DctParams bp{.q = args.size() > 2 ? parse_f(args[2], 2.0f) : 2.0f};
    auto a = codec::VolumeArchive::create(std::string(args[1]), vol->dims(), bp);
    if (!a) return std::unexpected(a.error());
    if (auto w = a->write_volume(vol->view()); !w) return std::unexpected(w.error());
    if (auto c = a->close(); !c) return std::unexpected(c.error());

    const Extent3 d = vol->dims();
    log(LogLevel::info, "ingested {} ({}x{}x{}) -> {} (DCT q={})", args[0], d.z, d.y, d.x, args[1], bp.q);
    return 0;
}

// `fenix ingest-zarr <zarr-root|url> <level> <z0> <y0> <x0> <D> <H> <W> <out.fxvol|.zarr>`
// Pull a dense [z0:z0+D, y0:y0+H, x0:x0+W] region from an OME-Zarr pyramid level (local path
// or s3://, http(s):// URL) and write it as .fxvol (transcoded) or a raw .zarr copy. Missing
// chunks = air. NEVER writes NRRD (foreign format, raw f32 on disk).
inline Expected<int> ingest_zarr(std::span<const std::string_view> args, Context&) {
    if (args.size() < 9) {
        log(LogLevel::error,
            "usage: fenix ingest-zarr <zarr-root|url> <level> <z0> <y0> <x0> <D> <H> <W> <out.fxvol|.zarr>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto pi = [](std::string_view s) { s64 v = 0; std::from_chars(s.data(), s.data() + s.size(), v); return v; };
    std::string root(args[0]);
    while (!root.empty() && root.back() == '/') root.pop_back();
    const std::string level(args[1]);
    const std::string lroot = root + "/" + level;  // pyramid level dir holds .zarray + chunks
    const Index3 origin{pi(args[2]), pi(args[3]), pi(args[4])};
    const Extent3 extent{pi(args[5]), pi(args[6]), pi(args[7])};
    const std::string outpath(args[8]);

    log(LogLevel::info, "ingest-zarr: {} level {} region z{}:{} y{}:{} x{}:{} -> {}", root, level,
        origin.z, origin.z + extent.z, origin.y, origin.y + extent.y, origin.x, origin.x + extent.x,
        outpath);

    // `.zarr` output: keep it native — raw-copy chunk bytes (preserves the source dtype, e.g. u8;
    // no f32 widening) and write fresh metadata for the sub-volume. Skips the f32 assemble below.
    if (outpath.size() > 5 && outpath.substr(outpath.size() - 5) == ".zarr") {
        if (auto r = copy_zarr_region_local(lroot, outpath, origin, extent); !r)
            return std::unexpected(r.error());
        log(LogLevel::info, "ingest-zarr: wrote local zarr {} ({}x{}x{} ZYX, raw chunk copy)", outpath,
            extent.z, extent.y, extent.x);
        return 0;
    }

    // u8 source (these scrolls are |u1) stays NATIVE u8 END-TO-END — NEVER widen the whole volume to f32
    // (a 640³ crop is 262 MB u8 vs 1 GB f32; a 2048³ is 8.6 GB vs 34 GB → OOM). This holds for BOTH the
    // NRRD path (raw u8 write) AND the .fxvol path: the archive encoder is templated on the source dtype
    // and widens one 64³ tile at a time inside the codec, so no 34 GB f32 buffer is ever assembled.
    {
        auto meta = read_zarray(lroot);
        if (!meta) return std::unexpected(meta.error());
        if (detail::dtype_size(meta->dtype) == 1) {
            auto v = read_zarr_region<u8>(lroot, origin, extent);
            if (!v) return std::unexpected(v.error());
            const Extent3 d = v->dims();
            auto a = codec::VolumeArchive::create(outpath, d, codec::DctParams{});
            if (!a) return std::unexpected(a.error());
            if (auto w = a->template write_volume<u8>(v->view()); !w) return std::unexpected(w.error());
            if (auto c = a->close(); !c) return std::unexpected(c.error());
            log(LogLevel::info, "ingest-zarr: wrote {} ({}x{}x{} ZYX, u8)", outpath, d.z, d.y, d.x);
            return 0;
        }
    }
    auto vol = read_zarr_region(lroot, origin, extent);  // f32 fallback (non-u8 source)
    if (!vol) return std::unexpected(vol.error());
    {
        auto a = codec::VolumeArchive::create(outpath, vol->dims(), codec::DctParams{});
        if (!a) return std::unexpected(a.error());
        if (auto w = a->write_volume(vol->view()); !w) return std::unexpected(w.error());
        if (auto c = a->close(); !c) return std::unexpected(c.error());
    }
    const Extent3 d = vol->dims();
    log(LogLevel::info, "ingest-zarr: wrote {} ({}x{}x{} ZYX)", outpath, d.z, d.y, d.x);
    return 0;
}

// `fenix export-scroll <zarr-root|url> <level> <out.fxvol> [q=8] [region=256] [z0 y0 x0 D H W]`
// Stream a whole OME-Zarr level into a .fxvol archive OUT-OF-CORE and RESUMABLY. Iterates the volume in
// `region`³ super-blocks (kept in RAM one at a time), reads each via the parallel zarr reader (missing
// chunks = air), writes its 64³ tiles, and COMMITS per region as a crash-safe checkpoint. RESUME: the
// archive's coverage tri-state is the bookmark — a region whose tiles are already present (Real/Zero) is
// skipped; a crash mid-region leaves those tiles ABSENT (COW) so a rerun redoes only the in-flight region.
// The optional [z0 y0 x0 D H W] limits the export to a sub-box of the (full-shape) archive.
inline Expected<int> export_scroll(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3) {
        log(LogLevel::error, "usage: fenix export-scroll <zarr-root|url> <level> <out.fxvol> [q=8] [region=256] [z0 y0 x0 D H W]");
        return err(Errc::invalid_argument, "missing args");
    }
    // The chunk-fetch parallel_for blocks on ~100 ms S3 round-trips; make idle OpenMP workers SLEEP at the
    // barrier rather than busy-spin (libomp spins for KMP_BLOCKTIME = 200 ms by default), which otherwise
    // pins every core to ~0% useful work for the whole network wait (measured: 16 cores vs <1). Set before
    // the first parallel_for (this is the first OMP use in the run); an explicit user env still wins.
    ::setenv("KMP_BLOCKTIME", "0", 0);
    ::setenv("OMP_WAIT_POLICY", "passive", 0);
    auto parse_f = [](std::string_view s, f32 d) { f32 v = d; std::from_chars(s.data(), s.data() + s.size(), v); return v; };
    std::string root(args[0]);
    while (!root.empty() && root.back() == '/') root.pop_back();
    const std::string lroot = root + "/" + std::string(args[1]);
    const std::string out(args[2]);
    const f32 q = args.size() > 3 ? parse_f(args[3], 8.0f) : 8.0f;
    s64 R = args.size() > 4 ? parse_i(args[4], 256) : 256;
    R = std::max<s64>(64, (R / 64) * 64);  // region must be a multiple of the 64³ tile
    // Batch crash-safe checkpoints. Each commit() COW-copies the touched root→L1→leaf page-table path, so a
    // PER-REGION commit strands ~one path of dead index nodes EVERY region — measured ~45% of a long export's
    // committed bytes were orphaned page-table nodes (verbatim `finalize` halved a 20.9 GiB live file to 11.2).
    // Committing every N spatially-adjacent regions lets them SHARE one COW of each node (the first region in
    // the batch copies it, the rest mutate it in place) → ~N× fewer orphans. Cost: a crash re-does ≤N regions
    // (the export is resumable from committed coverage) and ≤N regions of writes sit uncommitted in the mmap.
    // 64 ⇒ orphan bloat <1% at ~30 MiB of at-risk RAM. `commit=N` overrides (1 = the old per-region behavior).
    s64 commit_every = 64;
    for (auto s : args)
        if (s.size() > 7 && s.substr(0, 7) == "commit=") commit_every = std::max<s64>(1, parse_i(s.substr(7), 64));

    const s64 level_int = parse_i(args[1], 0);
    auto meta = read_zarray(lroot);
    if (!meta) return std::unexpected(meta.error());
    const Extent3 shape = meta->shape;
    // These CT scrolls are u8 (|u1). Keep the whole path NATIVE u8 — no f32 widening of the occupancy map
    // or region buffers (the DCT codec widens per 16³ block itself). u16/f32 sources would need the typed
    // path templated; reject them loudly rather than silently narrowing.
    if (detail::dtype_size(meta->dtype) != 1)
        return err(Errc::unsupported, "export-scroll: only u8 (|u1) sources supported, got " + meta->dtype);

    // Air-skip prefilter: load the COARSEST available pyramid level as an occupancy map (this is a masked
    // volume — most of the box is air). A super-region is skipped (left ABSENT = reads as air) unless some
    // coarse voxel in its footprint is non-zero. Robust: if no coarse level loads, process everything.
    Volume<u8> occ;
    s64 occ_scale = 1;
    for (s64 k = 5; k >= 1; --k) {
        auto m = read_zarray(root + "/" + std::to_string(level_int + k));
        if (!m) continue;
        auto o = read_zarr_region<u8>(root + "/" + std::to_string(level_int + k), {0, 0, 0}, m->shape);
        if (!o) continue;
        occ = std::move(*o);
        occ_scale = static_cast<s64>(1) << static_cast<u32>(k);  // 2^k voxels per coarse voxel
        log(LogLevel::info, "air-skip: occupancy from level {} ({}x{}x{}, scale {})", level_int + k,
            occ.dims().z, occ.dims().y, occ.dims().x, occ_scale);
        break;
    }
    auto occupied = [&](Index3 org, Extent3 ext) -> bool {
        if (occ.dims().z == 0) return true;  // no occupancy map → don't skip
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

    // Sub-box to export (default = whole volume). The archive is always created at the FULL shape.
    Index3 b0{0, 0, 0};
    Extent3 bd = shape;
    if (args.size() >= 11) {
        b0 = {parse_i(args[5], 0), parse_i(args[6], 0), parse_i(args[7], 0)};
        bd = {parse_i(args[8], shape.z), parse_i(args[9], shape.y), parse_i(args[10], shape.x)};
    }

    const bool resume = std::filesystem::exists(out);
    Expected<codec::VolumeArchive> ar = resume ? codec::VolumeArchive::open(out, true)
                                               : codec::VolumeArchive::create(out, shape, {.q = q});
    if (!ar) return std::unexpected(ar.error());
    codec::VolumeArchive a = std::move(*ar);
    if (resume && !(a.dims() == shape)) return err(Errc::invalid_argument, "resume: archive dims != zarr shape");

    const s64 T = codec::fxvol_chunk_side;  // 64
    auto ndiv = [](s64 n, s64 d) { return (n + d - 1) / d; };
    const s64 rz0 = b0.z / R, ry0 = b0.y / R, rx0 = b0.x / R;
    const s64 rz1 = ndiv(std::min(b0.z + bd.z, shape.z), R), ry1 = ndiv(std::min(b0.y + bd.y, shape.y), R), rx1 = ndiv(std::min(b0.x + bd.x, shape.x), R);
    const s64 total = (rz1 - rz0) * (ry1 - ry0) * (rx1 - rx0);
    log(LogLevel::info, "export-scroll {} L{} ({}x{}x{}) -> {} q={} region={} commit={}  {} regions  ({})", root,
        std::string(args[1]), shape.z, shape.y, shape.x, out, q, R, commit_every, total, resume ? "RESUME" : "fresh");

    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    auto t_last = t0;
    s64 skipped = 0, skipped_air = 0;
    u64 real_tiles = 0, zero_tiles = 0;

    // Build the fetch worklist up front: every occupied, not-yet-done region in the box. Resolving air-skip
    // + resume-skip HERE (instead of inside the fetch loop) means the network is never left idle while the
    // loop churns through long runs of skipped boundary regions — the prefetchers only ever see real work,
    // and `nwork` gives a stable ETA denominator (real regions remaining, not box regions).
    struct Work { Index3 org; Extent3 ext; };
    std::vector<Work> work;
    for (s64 rz = rz0; rz < rz1; ++rz)
        for (s64 ry = ry0; ry < ry1; ++ry)
            for (s64 rx = rx0; rx < rx1; ++rx) {
                const Index3 org{rz * R, ry * R, rx * R};
                const Extent3 ext{std::min(R, shape.z - org.z), std::min(R, shape.y - org.y), std::min(R, shape.x - org.x)};
                const ChunkCoord ft{org.z / T, org.y / T, org.x / T};  // a region commits atomically →
                if (a.coverage(0, ft) != codec::Coverage::Absent) { ++skipped; continue; }  // 1st tile present = done
                if (!occupied(org, ext)) { ++skipped_air; continue; }  // all-air per the coarse map → leave ABSENT
                work.push_back({org, ext});
            }
    const s64 nwork = static_cast<s64>(work.size());
    log(LogLevel::info, "  worklist: {} regions to fetch ({} air-skip, {} resume-skip, of {} in box)",
        nwork, skipped_air, skipped, total);

    // Prefetch pipeline. N producer threads pull regions off the worklist and fetch them (network-bound,
    // CPU idle) into a bounded queue; the single consumer (this thread) extracts + DCT-encodes + writes +
    // commits (CPU/disk-bound, network idle). Overlapping the download of region N+1 with the encode/commit
    // of region N keeps the link pinned near its ceiling instead of sawtoothing to 0 in every compute gap.
    // The cap is the *link*, not concurrency (measured ~8 MiB/s regardless of parallelism) — extra producers
    // only smooth the per-fetch tail (the 8-chunk barrier), they do not raise total throughput. The codec
    // archive is single-writer, so all write_chunk/commit stays on the consumer; only fetches are threaded.
    // Default tuned for pulling from S3 over a ~100 ms-RTT WAN link: each connection is latency-limited, so
    // ~64 concurrent GETs (8 producers × 8 chunks/region) are needed to fill a fast pipe (measured ~640 of
    // 660 Mbit). Idle producers cost ~nothing (passive OMP wait). Tune down for low-RTT/in-region or a weak
    // router; up for very-high-RTT links. RAM bound = (nprod+2) regions × region³ × 1 B (native u8).
    int nprod = 8;
    if (const char* e = std::getenv("FENIX_EXPORT_PREFETCH")) { const int v = std::atoi(e); if (v >= 1 && v <= 32) nprod = v; }
    const usize qcap = static_cast<usize>(nprod) + 2;  // bounded RAM: ~qcap * region³ * 1B u8 (256³ ≈ 16 MiB each)
    log(LogLevel::info, "  prefetch: {} producer(s), queue depth {} (set FENIX_EXPORT_PREFETCH to tune)", nprod, qcap);

    struct Fetched { s64 idx; Volume<u8> vol; };
    std::mutex mtx;
    std::condition_variable cv_push, cv_pop;
    std::deque<Fetched> queue;
    std::atomic<s64> cursor{0};  // next worklist index to claim
    s64 producers_live = nprod;
    bool cancel = false;
    Error fetch_err;
    bool has_err = false;

    auto producer = [&] {
        for (;;) {
            const s64 i = cursor.fetch_add(1);
            if (i >= nwork) break;
            { std::unique_lock lk(mtx); if (cancel) break; }
            const Work w = work[static_cast<usize>(i)];
            // Ride through transient network blips (DNS/5xx/timeout) with capped exponential backoff — a
            // multi-hour unattended export must not die on one bad GET; give up only on a sustained outage
            // (the job is resumable, so a rerun continues from the committed coverage regardless).
            Expected<Volume<u8>> reg = read_zarr_region<u8>(lroot, w.org, w.ext);
            for (int attempt = 1; !reg && attempt <= 20; ++attempt) {
                { std::unique_lock lk(mtx); if (cancel) break; }
                const int backoff = std::min(30, 1 << std::min(attempt, 5));  // 2,4,8,16,30,30,...
                log(LogLevel::warn, "region z{} y{} x{} read failed: {} — retry {}/20 in {}s", w.org.z, w.org.y, w.org.x,
                    reg.error().message, attempt, backoff);
                std::this_thread::sleep_for(std::chrono::seconds(backoff));
                reg = read_zarr_region<u8>(lroot, w.org, w.ext);
            }
            if (!reg) {  // sustained outage → signal abort; consumer drains what's buffered, then reports it
                std::unique_lock lk(mtx);
                if (!has_err) { fetch_err = reg.error(); has_err = true; }
                cancel = true;
                cv_pop.notify_all();
                break;
            }
            std::unique_lock lk(mtx);
            cv_push.wait(lk, [&] { return queue.size() < qcap || cancel; });
            if (cancel) break;
            queue.push_back({i, std::move(*reg)});
            cv_pop.notify_one();
        }
        std::unique_lock lk(mtx);
        if (--producers_live == 0) cv_pop.notify_all();
    };

    std::vector<std::thread> pool;
    pool.reserve(static_cast<usize>(nprod));
    for (int p = 0; p < nprod; ++p) pool.emplace_back(producer);

    s64 done = 0;
    s64 since_commit = 0;  // regions written into the current (uncommitted) COW transaction
    Expected<void> consume_err;
    std::vector<u8> block(static_cast<usize>(T * T * T));
    for (;;) {
        Fetched f;
        {
            std::unique_lock lk(mtx);
            cv_pop.wait(lk, [&] { return !queue.empty() || producers_live == 0; });
            if (queue.empty()) break;  // all producers done and nothing left to drain
            f = std::move(queue.front());
            queue.pop_front();
            cv_push.notify_one();
        }
        if (const auto nowt = clk::now(); std::chrono::duration<f64>(nowt - t_last).count() >= 15.0 || done == 0) {
            t_last = nowt;
            const f64 el = std::chrono::duration<f64>(nowt - t0).count();
            const f64 frac = static_cast<f64>(done) / static_cast<f64>(nwork ? nwork : 1);
            usize qd;
            { std::unique_lock lk(mtx); qd = queue.size(); }
            log(LogLevel::info, "  {}/{} fetched ({:.1f}%) {:.0f}s ETA {:.0f}s | buffered {}/{} | air-skip {} resume-skip {} | real {} zero {} tiles | {:.1f}MiB",
                done, nwork, 100.0 * frac, el, frac > 0 ? el * (1.0 - frac) / frac : 0.0, qd, qcap, skipped_air, skipped,
                real_tiles, zero_tiles, static_cast<f64>(a.committed_size()) / (1024.0 * 1024.0));
        }
        const Index3 org = work[static_cast<usize>(f.idx)].org;
        const Extent3 ext = work[static_cast<usize>(f.idx)].ext;
        auto rv = f.vol.view();
        bool ok = true;
        for (s64 tz = 0; ok && tz < ndiv(ext.z, T); ++tz)
            for (s64 ty = 0; ok && ty < ndiv(ext.y, T); ++ty)
                for (s64 tx = 0; ok && tx < ndiv(ext.x, T); ++tx) {
                    for (s64 z = 0; z < T; ++z)  // extract a 64³ tile, edge-replicating at the box edge
                        for (s64 y = 0; y < T; ++y)
                            for (s64 x = 0; x < T; ++x)
                                block[static_cast<usize>((z * T + y) * T + x)] =
                                    rv(std::min(tz * T + z, ext.z - 1), std::min(ty * T + y, ext.y - 1), std::min(tx * T + x, ext.x - 1));
                    const ChunkCoord tc{org.z / T + tz, org.y / T + ty, org.x / T + tx};
                    if (auto w = a.write_chunk(0, tc, block); !w) { consume_err = std::unexpected(w.error()); ok = false; break; }
                    (a.coverage(0, tc) == codec::Coverage::Real ? real_tiles : zero_tiles)++;
                }
        if (ok && ++since_commit >= commit_every) {  // batched crash-safe checkpoint (see commit_every above)
            if (auto c = a.commit(); !c) { consume_err = std::unexpected(c.error()); ok = false; }
            since_commit = 0;
        }
        ++done;
        if (!ok) { std::unique_lock lk(mtx); cancel = true; cv_push.notify_all(); break; }
    }

    for (auto& th : pool) th.join();
    if (has_err) return std::unexpected(fetch_err);    // sustained fetch outage → abort; rerun resumes
    if (!consume_err) return std::unexpected(consume_err.error());  // write/commit failure
    if (auto c = a.close(); !c) return std::unexpected(c.error());
    log(LogLevel::info, "export-scroll done: {} regions ({} air-skipped, {} resume-skipped), real {} / zero {} tiles, {:.1f} MiB",
        total, skipped_air, skipped, real_tiles, zero_tiles, static_cast<f64>(a.committed_size()) / (1024.0 * 1024.0));
    return 0;
}


// `fenix transcode <in.fxvol> <out.fxvol> <q>` — re-encode a .fxvol at a new DCT quality WITHOUT a NRRD
// round-trip. Decodes LOD0 to a dense volume in memory and re-encodes at q. For a probability field the
// dense buffer is f32 (that is what the prob volume genuinely is); no 34 GB NRRD lands on disk. Note the
// re-encode is lossy-on-lossy (the input is already quantized) — fine for a compact transfer copy.
inline Expected<int> transcode_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 3) {
        log(LogLevel::error, "usage: fenix transcode <in.fxvol> <out.fxvol> <q>");
        return err(Errc::invalid_argument, "missing args");
    }
    f32 q = 8.0f;
    std::from_chars(args[2].data(), args[2].data() + args[2].size(), q);
    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    auto vol = a->read_volume(0);  // decode LOD0 dense (f32 prob field)
    if (!vol) return std::unexpected(vol.error());
    const Extent3 d = vol->dims();
    auto out = codec::VolumeArchive::create(std::string(args[1]), d, codec::DctParams{.q = q});
    if (!out) return std::unexpected(out.error());
    if (auto w = out->write_volume(vol->view()); !w) return std::unexpected(w.error());
    if (auto c = out->close(); !c) return std::unexpected(c.error());
    log(LogLevel::info, "transcode {} -> {} at q={} ({}x{}x{})", args[0], args[1], q, d.z, d.y, d.x);
    return 0;
}

// `fenix finalize <in.fxvol> <out.fxvol>` — repack into the SEALED coarse-first, front-loaded-index form.
inline Expected<int> finalize_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix finalize <in.fxvol> <out.fxvol>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    if (auto r = a->finalize(std::string(args[1])); !r) return std::unexpected(r.error());
    log(LogLevel::info, "finalized {} -> {} (sealed: coarse-first, front-loaded index)", args[0], args[1]);
    return 0;
}

// `fenix fxinfo <in.fxvol>` — inspect dims, LOD pyramid, coverage tri-state counts, size, compression ratio.
inline Expected<int> info_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 1) {
        log(LogLevel::error, "usage: fenix fxinfo <in.fxvol>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto a = codec::VolumeArchive::open(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    const Extent3 d = a->dims();
    std::error_code ec;
    const u64 fsz = std::filesystem::file_size(std::string(args[0]), ec);
    const f64 voxels = static_cast<f64>(d.z) * static_cast<f64>(d.y) * static_cast<f64>(d.x);
    const u64 csz = a->committed_size();  // actual data bytes (the file may be fallocate-padded if not closed)
    std::printf("fxvol %s\n  dims (ZYX) %lldx%lldx%lld   q=%.2f   LODs=%u   %s\n  data %.2f MiB (file %.2f MiB)   ratio8 %.1fx\n",
                std::string(args[0]).c_str(), (long long)d.z, (long long)d.y, (long long)d.x,
                static_cast<double>(a->params().q), a->nlods(), a->data_offset() ? "SEALED" : "live",
                static_cast<double>(csz) / (1024.0 * 1024.0), static_cast<double>(fsz) / (1024.0 * 1024.0),
                voxels / static_cast<f64>(csz ? csz : 1));
    for (s64 lod = 0; lod < static_cast<s64>(a->nlods()); ++lod) {
        const Extent3 ld = a->dims_at(lod);
        const ChunkCoord ce = a->chunk_extent(lod);
        u64 real = 0, zero = 0, absent = 0;
        for (s64 cz = 0; cz < ce.z; ++cz)
            for (s64 cy = 0; cy < ce.y; ++cy)
                for (s64 cx = 0; cx < ce.x; ++cx)
                    switch (a->coverage(lod, {cz, cy, cx})) {
                        case codec::Coverage::Real: ++real; break;
                        case codec::Coverage::Zero: ++zero; break;
                        default: ++absent; break;
                    }
        std::printf("  LOD%-2lld %5lldx%-5lld x%-5lld  chunks %lldx%lldx%lld   real %llu  zero %llu  absent %llu\n",
                    (long long)lod, (long long)ld.z, (long long)ld.y, (long long)ld.x, (long long)ce.z,
                    (long long)ce.y, (long long)ce.x, (unsigned long long)real, (unsigned long long)zero,
                    (unsigned long long)absent);
    }
    return 0;
}

// `fenix compare <a.nrrd> <b.nrrd>` — PSNR / MAE / max-abs-error between two volumes (roundtrip check).
inline Expected<int> compare_vol(std::span<const std::string_view> args, Context&) {
    if (args.size() < 2) {
        log(LogLevel::error, "usage: fenix compare <a.nrrd> <b.nrrd>");
        return err(Errc::invalid_argument, "missing args");
    }
    auto a = read_nrrd(std::string(args[0]));
    if (!a) return std::unexpected(a.error());
    auto b = read_nrrd(std::string(args[1]));
    if (!b) return std::unexpected(b.error());
    if (!(a->dims() == b->dims())) return err(Errc::invalid_argument, "dims mismatch");
    const Extent3 d = a->dims();
    auto av = a->view(), bv = b->view();
    f64 sse = 0, sae = 0, mx = 0;
    for (s64 z = 0; z < d.z; ++z)
        for (s64 y = 0; y < d.y; ++y)
            for (s64 x = 0; x < d.x; ++x) {
                const f64 e = std::abs(static_cast<f64>(av(z, y, x)) - bv(z, y, x));
                sse += e * e;
                sae += e;
                mx = std::max(mx, e);
            }
    const f64 n = static_cast<f64>(d.z) * static_cast<f64>(d.y) * static_cast<f64>(d.x);
    const f64 mse = sse / n;
    std::printf("compare %s vs %s (%lldx%lldx%lld)\n  PSNR %.2f dB   MAE %.4f   max-abs %.4f\n",
                std::string(args[0]).c_str(), std::string(args[1]).c_str(), (long long)d.z, (long long)d.y,
                (long long)d.x, mse > 0 ? 10.0 * std::log10(255.0 * 255.0 / mse) : 99.0, sae / n, mx);
    return 0;
}

// `fenix fxupgrade <archive.fxvol> [dtype=u8]` — in-place v4 → v5 upgrade. v4→v5 changed ONLY the semantics
// of a formerly-reserved superblock byte (offset 28) into a source-dtype tag; the byte layout is otherwise
// identical (DCT blobs, page table, LOD pyramid untouched). So the upgrade just rewrites the two 4 KiB
// superblocks: version 4→5, stamp the dtype byte (default u8 — every scroll export is u8), refresh the crc.
// Milliseconds regardless of archive size; no re-encode. Idempotent (a v5 file is left unchanged).
inline Expected<int> fxupgrade(std::span<const std::string_view> args, Context&) {
    if (args.empty()) {
        log(LogLevel::error, "usage: fenix fxupgrade <archive.fxvol> [dtype=u8|u16|u32|s8|s16|s32|f16|f32]");
        return err(Errc::invalid_argument, "missing archive path");
    }
    const std::string path(args[0]);
    codec::DType dt = codec::DType::u8;
    if (args.size() >= 2) {
        const std::string_view s = args[1];
        if (s == "u8") dt = codec::DType::u8;
        else if (s == "u16") dt = codec::DType::u16;
        else if (s == "u32") dt = codec::DType::u32;
        else if (s == "s8") dt = codec::DType::s8;
        else if (s == "s16") dt = codec::DType::s16;
        else if (s == "s32") dt = codec::DType::s32;
        else if (s == "f16") dt = codec::DType::f16;
        else if (s == "f32") dt = codec::DType::f32;
        else return err(Errc::invalid_argument, "bad dtype (u8|u16|u32|s8|s16|s32|f16|f32)");
    }

    std::FILE* f = std::fopen(path.c_str(), "r+b");
    if (!f) return err(Errc::not_found, "cannot open " + path);
    using namespace codec::detail;
    int patched = 0, already = 0;
    for (int slot = 0; slot < 2; ++slot) {
        u8 buf[kFxSuper];
        if (std::fseek(f, static_cast<long>(slot * kFxSuper), SEEK_SET) != 0) { std::fclose(f); return err(Errc::io_error, "seek"); }
        if (std::fread(buf, 1, kFxSuper, f) != kFxSuper) { std::fclose(f); return err(Errc::io_error, "short read"); }
        u32 magic, ver;
        std::memcpy(&magic, buf + 0, 4);
        std::memcpy(&ver, buf + 4, 4);
        if (magic != codec::fxvol_magic) continue;  // not a superblock slot (e.g. empty B slot) — skip
        if (ver == codec::fxvol_version) { ++already; continue; }
        if (ver != 4) { std::fclose(f); return err(Errc::unsupported, "only v4 → v5 upgrade is supported"); }
        const u32 v5 = codec::fxvol_version;
        std::memcpy(buf + 4, &v5, 4);
        const u32 dtw = static_cast<u32>(static_cast<u8>(dt));
        std::memcpy(buf + 28, &dtw, 4);
        const u32 crc = crc32c(buf, kFxSbCrcLen);
        std::memcpy(buf + kFxSbCrcLen, &crc, 4);
        if (std::fseek(f, static_cast<long>(slot * kFxSuper), SEEK_SET) != 0) { std::fclose(f); return err(Errc::io_error, "seek"); }
        if (std::fwrite(buf, 1, kFxSuper, f) != kFxSuper) { std::fclose(f); return err(Errc::io_error, "write"); }
        ++patched;
    }
    std::fflush(f);
    std::fclose(f);
    if (patched == 0 && already > 0) {
        log(LogLevel::info, "fxupgrade: {} already v{} (no change)", path, codec::fxvol_version);
        return 0;
    }
    if (patched == 0) return err(Errc::decode_error, "no valid superblock found");
    log(LogLevel::info, "fxupgrade: {} → v{} ({} superblock(s) patched, dtype={})", path, codec::fxvol_version,
        patched, static_cast<int>(static_cast<u8>(dt)));
    return 0;
}

}  // namespace fenix::io

FENIX_REGISTER_STAGE(ingest, "ingest a NRRD volume into a .fxvol archive", ::fenix::io::ingest)

namespace {
[[maybe_unused]] const int fenix_stage_ingest_zarr = ::fenix::register_stage(
    ::fenix::Stage{"ingest-zarr", "pull an OME-Zarr region (local/s3/http) into .fxvol/.nrrd",
                   ::fenix::io::ingest_zarr});
[[maybe_unused]] const int fenix_stage_fxupgrade = ::fenix::register_stage(
    ::fenix::Stage{"fxupgrade", "upgrade a .fxvol v4 → v5 in place (adds the source-dtype tag)",
                   ::fenix::io::fxupgrade});
[[maybe_unused]] const int fenix_stage_export_scroll = ::fenix::register_stage(
    ::fenix::Stage{"export-scroll", "stream a whole OME-Zarr level into .fxvol (out-of-core, resumable)", ::fenix::io::export_scroll});
[[maybe_unused]] const int fenix_stage_transcode = ::fenix::register_stage(
    ::fenix::Stage{"transcode", "re-encode a .fxvol at a new DCT quality (no NRRD round-trip)", ::fenix::io::transcode_vol});
[[maybe_unused]] const int fenix_stage_finalize = ::fenix::register_stage(
    ::fenix::Stage{"finalize", "repack a .fxvol into the SEALED coarse-first form", ::fenix::io::finalize_vol});
[[maybe_unused]] const int fenix_stage_fxinfo = ::fenix::register_stage(
    ::fenix::Stage{"fxinfo", "inspect a .fxvol (dims, LODs, coverage, size, ratio)", ::fenix::io::info_vol});
[[maybe_unused]] const int fenix_stage_compare = ::fenix::register_stage(
    ::fenix::Stage{"compare", "PSNR/MAE/max-abs between two NRRD volumes", ::fenix::io::compare_vol});
}  // namespace
