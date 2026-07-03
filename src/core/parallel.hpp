// core/parallel.hpp — the data-parallel primitive (OpenMP), with a serial fallback when
// OpenMP isn't present. A first-party async thread pool for IO pipelines lives separately.
#pragma once

#include "core/types.hpp"

#include <atomic>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace fenix {

namespace detail {
// Effective CPU budget for this process. In a container, nproc / omp_get_num_procs() report the HOST
// core count (e.g. 256), but the cgroup CPU quota may be far smaller (e.g. 27). Spawning one OMP thread
// per host core then oversubscribes the quota ~10× — every parallel_for pays barrier/scheduling churn on
// cores it can't actually use. Read the real quota from cgroup v2 (cpu.max = "<quota> <period>", ceil of
// quota/period) or v1 (cfs_quota_us/cfs_period_us); fall back to hardware_concurrency.
inline int cgroup_cpu_budget() {
    auto read_pair = [](const char* path, long& a, long& b) -> bool {
        std::FILE* f = std::fopen(path, "r");
        if (!f) return false;
        char buf[128] = {};
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        std::fclose(f);
        if (n == 0) return false;
        std::string_view sv(buf, n);
        // parse first two whitespace-separated tokens; token 0 may be "max"
        size_t i = 0;
        auto tok = [&](std::string_view& out) {
            while (i < sv.size() && (sv[i] == ' ' || sv[i] == '\n' || sv[i] == '\t')) ++i;
            const size_t s = i;
            while (i < sv.size() && sv[i] != ' ' && sv[i] != '\n' && sv[i] != '\t') ++i;
            out = sv.substr(s, i - s);
            return !out.empty();
        };
        std::string_view t0, t1;
        if (!tok(t0)) return false;
        if (t0 == "max") {
            a = -1;
            b = 1;
            return true;
        }
        long va = 0;
        if (std::from_chars(t0.data(), t0.data() + t0.size(), va).ec != std::errc{}) return false;
        a = va;
        if (tok(t1)) {
            long vb = 0;
            std::from_chars(t1.data(), t1.data() + t1.size(), vb);
            b = vb ? vb : 1;
        } else
            b = 1;
        return true;
    };
    long quota = -1, period = 1;
    if (read_pair("/sys/fs/cgroup/cpu.max", quota, period) && quota > 0) {
        const int c = static_cast<int>((quota + period - 1) / period);  // ceil
        if (c >= 1) return c;
    }
    // cgroup v1: two separate files
    long q1 = -1, p1 = 100000, dummy = 0;
    if (read_pair("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", q1, dummy) && q1 > 0 &&
        read_pair("/sys/fs/cgroup/cpu/cpu.cfs_period_us", p1, dummy) && p1 > 0) {
        const int c = static_cast<int>((q1 + p1 - 1) / p1);
        if (c >= 1) return c;
    }
    const unsigned hc = std::thread::hardware_concurrency();
    return hc ? static_cast<int>(hc) : 1;
}
}  // namespace detail

// Return the thread budget this process should use for CPU-bound parallel regions: the min of the cgroup
// CPU quota and the host core count, overridable by FENIX_THREADS (or OMP_NUM_THREADS). Cached.
inline int cpu_budget() {
    static const int n = [] {
        if (const char* e = std::getenv("FENIX_THREADS")) {
            const int v = std::atoi(e);
            if (v > 0) return v;
        }
        if (const char* e = std::getenv("OMP_NUM_THREADS")) {
            const int v = std::atoi(e);
            if (v > 0) return v;
        }
        return detail::cgroup_cpu_budget();
    }();
    return n;
}

// Clamp OpenMP's default team size to the real CPU budget (call ONCE at startup, before any parallel_for).
// Without this, libomp defaults to omp_get_num_procs() = the HOST core count and oversubscribes a
// CPU-quota-limited container. Idempotent; honours an explicit FENIX_THREADS/OMP_NUM_THREADS.
inline void init_thread_limits() {
#if defined(_OPENMP)
    omp_set_num_threads(cpu_budget());
    // Disallow OMP-in-OMP explicitly (libomp's default, pinned here against stray
    // OMP_MAX_ACTIVE_LEVELS env). NOTE this only guards OpenMP's own thread tree: a
    // std::thread hitting a parallel region is a fresh LEVEL-0 initial thread, allowed a
    // full-width team — that topology (feed workers) needs SerialRegion below instead.
    omp_set_max_active_levels(1);
#endif
}

// When set (thread-local), `parallel_for`/`parallel_for_dynamic` run SERIALLY on the calling thread.
// This is the anti-nesting switch: an outer parallel loop over independent work units (e.g. the tiled
// tracer over disjoint tiles) wraps each unit's body in a `SerialRegion` so the per-unit kernels
// (structure tensor, gaussian blur, ARAP) do NOT spawn nested OpenMP regions — nesting oversubscribes
// the cores (measured ~10× slowdown). A thread-local flag is cheaper and more robust than fiddling with
// OMP_NESTED / max_active_levels, and it composes: the outer loop parallelizes ONE level, everything
// below it is serial on that worker.
inline thread_local bool g_parallel_serial = false;

// RAII: force serial execution of nested parallel loops for the current scope (restores on exit, so it
// nests correctly and leaves the worker thread's flag clean for reuse by later parallel regions).
struct SerialRegion {
    bool prev_;
    SerialRegion() : prev_(g_parallel_serial) { g_parallel_serial = true; }
    ~SerialRegion() { g_parallel_serial = prev_; }
    SerialRegion(const SerialRegion&) = delete;
    SerialRegion& operator=(const SerialRegion&) = delete;
};

// parallel_for over [begin, end): calls body(i) for each i, in OpenMP-scheduled chunks.
// Body must be thread-safe across distinct i. Determinism is not guaranteed (fast-math,
// scheduling) — that is accepted project-wide. Runs serial if `g_parallel_serial` is set.
//
// max_threads caps the worker count for this loop (0 = use all available). This is the
// throttle for IO fan-out: remote chunk fetches over one endpoint want a bounded number of
// concurrent connections (too many self-congest and trip the per-transfer stall watchdog),
// not one connection per core. CPU-bound loops leave it 0.
template <class Body> void parallel_for(s64 begin, s64 end, Body&& body, int max_threads = 0) {
#if defined(_OPENMP)
    if (g_parallel_serial) {
        for (s64 i = begin; i < end; ++i) body(i);
    } else {
        // Always bound the team to the CPU budget (cgroup quota), even when the caller passes 0 — this
        // makes every parallel_for container-safe regardless of whether init_thread_limits() ran in this
        // TU (tests, fuzz, bench have their own mains). An explicit max_threads>0 (IO fan-out) caps further.
        const int nt = max_threads > 0 ? (max_threads < cpu_budget() ? max_threads : cpu_budget()) : cpu_budget();
        if (max_threads > 0) {
#pragma omp parallel for schedule(dynamic) num_threads(nt)
            for (s64 i = begin; i < end; ++i) body(i);
        } else {
#pragma omp parallel for schedule(static) num_threads(nt)
            for (s64 i = begin; i < end; ++i) body(i);
        }
    }
#else
    (void)max_threads;
    for (s64 i = begin; i < end; ++i) body(i);
#endif
}

// Like parallel_for but DYNAMIC scheduling (chunk=1) — for a few heavy, WILDLY-UNEVEN work items where
// static scheduling starves cores (e.g. the tiled tracer: dense papyrus tiles dwarf edge tiles). Each
// body() should be coarse (a whole tile), so the per-item dispatch cost is negligible. Container-safe
// (bounded to cpu_budget) and honours `g_parallel_serial`.
template <class Body> void parallel_for_dynamic(s64 begin, s64 end, Body&& body) {
#if defined(_OPENMP)
    if (g_parallel_serial) {
        for (s64 i = begin; i < end; ++i) body(i);
    } else {
        const int nt = cpu_budget();
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
        for (s64 i = begin; i < end; ++i) body(i);
    }
#else
    for (s64 i = begin; i < end; ++i) body(i);
#endif
}

// parallel_for for I/O-BOUND fan-out (remote chunk fetch): the workers spend nearly all their time
// blocked on the network, not on CPU, so the pool is sized to `threads` DIRECTLY and NOT clamped to
// cpu_budget() — a low CPU quota (e.g. a 13-CPU container) must not throttle the number of concurrent
// S3 connections, or throughput collapses to ~cpu_budget parallel transfers. `threads` is the hard
// ceiling (too many self-congest one endpoint and trip the stall watchdog); pick it for the endpoint,
// not the core count.
//
// Uses std::thread, NOT OpenMP: libomp derives its own max team size from the cgroup/affinity at
// startup and will silently clamp a `num_threads(64)` clause back to a handful (observed: a team of 1
// on a 13-CPU-quota container, "Cannot form a team with 64 threads, using 1 instead"). A raw thread
// pool over-subscribes intentionally — correct for I/O, and immune to libomp's CPU-bound heuristics.
// Honours g_parallel_serial. Work is claimed via a shared atomic cursor (dynamic, uneven latency).
template <class Body> void parallel_for_io(s64 begin, s64 end, int threads, Body&& body) {
    if (g_parallel_serial || end - begin <= 1) {
        for (s64 i = begin; i < end; ++i) body(i);
        return;
    }
    int nt = threads > 0 ? threads : cpu_budget();
    const s64 span = end - begin;
    if (static_cast<s64>(nt) > span) nt = static_cast<int>(span);
    std::atomic<s64> cursor{begin};
    auto worker = [&] {
        for (;;) {
            const s64 i = cursor.fetch_add(1, std::memory_order_relaxed);
            if (i >= end) break;
            body(i);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(static_cast<usize>(nt - 1));
    for (int t = 0; t < nt - 1; ++t) pool.emplace_back(worker);
    worker();  // the calling thread is one of the workers
    for (auto& th : pool) th.join();
}

// Convenience: parallelize over the z-slices of a ZYX volume extent (the common pattern).
template <class Body> void parallel_for_z(Extent3 dims, Body&& body) {
    parallel_for(0, dims.z, std::forward<Body>(body));
}

}  // namespace fenix
