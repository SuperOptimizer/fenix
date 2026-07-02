// view/view.hpp — module umbrella: the Qt-free viewer engine (slice/oblique/composite
// rendering + prefetch over .fxvol). The Qt shell in gui/ builds on this; headless
// consumers (CLI slice export, tests, benches) use it directly. See view/CLAUDE.md.
#pragma once

#include "core/core.hpp"

#include "view/composite.hpp"
#include "view/prefetch.hpp"
#include "view/sampler.hpp"
#include "view/slice_engine.hpp"
