// core/parallel.hpp — the data-parallel primitive (OpenMP), with a serial fallback when
// OpenMP isn't present. A first-party async thread pool for IO pipelines lives separately.
#pragma once

#include "core/types.hpp"

#include <functional>

namespace fenix {

// parallel_for over [begin, end): calls body(i) for each i, in OpenMP-scheduled chunks.
// Body must be thread-safe across distinct i. Determinism is not guaranteed (fast-math,
// scheduling) — that is accepted project-wide.
//
// max_threads caps the worker count for this loop (0 = use all available). This is the
// throttle for IO fan-out: remote chunk fetches over one endpoint want a bounded number of
// concurrent connections (too many self-congest and trip the per-transfer stall watchdog),
// not one connection per core. CPU-bound loops leave it 0.
template <class Body>
void parallel_for(s64 begin, s64 end, Body&& body, int max_threads = 0) {
#if defined(_OPENMP)
    if (max_threads > 0) {
#pragma omp parallel for schedule(dynamic) num_threads(max_threads)
        for (s64 i = begin; i < end; ++i) body(i);
    } else {
#pragma omp parallel for schedule(static)
        for (s64 i = begin; i < end; ++i) body(i);
    }
#else
    (void)max_threads;
    for (s64 i = begin; i < end; ++i) body(i);
#endif
}

// Convenience: parallelize over the z-slices of a ZYX volume extent (the common pattern).
template <class Body>
void parallel_for_z(Extent3 dims, Body&& body) {
    parallel_for(0, dims.z, std::forward<Body>(body));
}

}  // namespace fenix
