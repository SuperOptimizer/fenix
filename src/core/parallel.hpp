// core/parallel.hpp — the data-parallel primitive (OpenMP), with a serial fallback when
// OpenMP isn't present. A first-party async thread pool for IO pipelines lives separately.
#pragma once

#include "core/types.hpp"

#include <functional>

namespace fenix {

// parallel_for over [begin, end): calls body(i) for each i, in OpenMP-scheduled chunks.
// Body must be thread-safe across distinct i. Determinism is not guaranteed (fast-math,
// scheduling) — that is accepted project-wide.
template <class Body>
void parallel_for(s64 begin, s64 end, Body&& body) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
    for (s64 i = begin; i < end; ++i) body(i);
#else
    for (s64 i = begin; i < end; ++i) body(i);
#endif
}

// Convenience: parallelize over the z-slices of a ZYX volume extent (the common pattern).
template <class Body>
void parallel_for_z(Extent3 dims, Body&& body) {
    parallel_for(0, dims.z, std::forward<Body>(body));
}

}  // namespace fenix
