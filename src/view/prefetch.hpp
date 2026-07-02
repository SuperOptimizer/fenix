// view/prefetch.hpp — viewport-prioritized async chunk warmer. The archive's SIEVE block
// cache has no readahead of its own; this pulls the 64³ tiles a viewer is ABOUT to need
// (neighbouring slices, the zoomed-out ring) through block16() on background threads so
// scrolling hits warm cache. begin_batch() drops stale queued work when the viewport
// moves — prefetch is best-effort, never a backlog. Decode errors are swallowed here (a
// prefetch must never fail a render; the render path reports its own errors).
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

namespace fenix::view {

class Prefetcher {
public:
    explicit Prefetcher(const codec::VolumeArchive& a, int threads = 2) : arch_(a) {
        threads = std::clamp(threads, 1, 8);
        for (int i = 0; i < threads; ++i)
            workers_.emplace_back([this](std::stop_token st) { work_(st); });
    }
    ~Prefetcher() {
        {
            std::lock_guard lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
    }
    Prefetcher(const Prefetcher&) = delete;
    Prefetcher& operator=(const Prefetcher&) = delete;

    // The viewport moved: drop queued (not-yet-started) work and start a fresh dedupe epoch.
    void begin_batch() {
        std::lock_guard lk(m_);
        heap_.clear();
        seen_.clear();
    }

    // Queue one 64³ tile at `lod` for warming; higher priority pops first.
    void request_tile(s64 lod, ChunkCoord tile, f32 priority) {
        const ChunkCoord ce = arch_.chunk_extent(lod);
        if (tile.z < 0 || tile.y < 0 || tile.x < 0 || tile.z >= ce.z || tile.y >= ce.y || tile.x >= ce.x) return;
        const u64 key = (static_cast<u64>(lod) << 54) | (static_cast<u64>(tile.z) << 36) |
                        (static_cast<u64>(tile.y) << 18) | static_cast<u64>(tile.x);
        {
            std::lock_guard lk(m_);
            if (!seen_.insert(key).second) return;
            heap_.push_back({priority, lod, tile});
            std::push_heap(heap_.begin(), heap_.end(), by_pri_);
        }
        cv_.notify_one();
    }

    // Block until all queued work has been executed (tests / synchronous warmup).
    void drain() {
        std::unique_lock lk(m_);
        idle_cv_.wait(lk, [this] { return heap_.empty() && active_ == 0; });
    }

    [[nodiscard]] u64 warmed() const { return warmed_.load(std::memory_order_relaxed); }

private:
    struct Job {
        f32 pri;
        s64 lod;
        ChunkCoord tile;
    };
    static constexpr auto by_pri_ = [](const Job& a, const Job& b) { return a.pri < b.pri; };

    void work_(const std::stop_token& st) {
        constexpr s64 kBlocksPerTile = codec::fxvol_chunk_side / codec::kDctN;
        while (true) {
            Job j;
            {
                std::unique_lock lk(m_);
                cv_.wait(lk, [this] { return stop_ || !heap_.empty(); });
                if (stop_ && heap_.empty()) return;
                if (stop_) return;
                std::pop_heap(heap_.begin(), heap_.end(), by_pri_);
                j = heap_.back();
                heap_.pop_back();
                ++active_;
            }
            // Warming any 16³ sub-block decodes + caches the whole 64³ tile.
            (void)arch_.block16(j.lod, {j.tile.z * kBlocksPerTile, j.tile.y * kBlocksPerTile,
                                        j.tile.x * kBlocksPerTile});
            warmed_.fetch_add(1, std::memory_order_relaxed);
            {
                std::lock_guard lk(m_);
                --active_;
            }
            idle_cv_.notify_all();
            if (st.stop_requested()) return;
        }
    }

    const codec::VolumeArchive& arch_;
    std::vector<Job> heap_;
    std::unordered_set<u64> seen_;
    std::mutex m_;
    std::condition_variable cv_, idle_cv_;
    int active_ = 0;
    bool stop_ = false;
    std::atomic<u64> warmed_{0};
    std::vector<std::jthread> workers_;  // last member: joins before the rest tears down
};

}  // namespace fenix::view
