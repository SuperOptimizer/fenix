// fenix.hpp — the umbrella header. Transitively includes every module so that
// apps/driver.cpp (the single translation unit) pulls in the whole program. Each module
// self-registers its stage(s) on include. gui is firewalled behind FENIX_GUI (Qt/VTK).
#pragma once

#include "core/core.hpp"

#include "annotate/annotate.hpp"
#include "codec/codec.hpp"
#include "eval/eval.hpp"
#include "flatten/flatten.hpp"
#include "geom/geom.hpp"
#include "io/io.hpp"
#include "ml/ml.hpp"
#include "postproc/postproc.hpp"
#include "predictions/predictions.hpp"
#include "preprocess/preprocess.hpp"
#include "render/render.hpp"
#include "segment/segment.hpp"
#include "topo/topo.hpp"
#include "view/view.hpp"
#include "winding/winding.hpp"

#ifdef FENIX_GUI
#include "gui/gui.hpp"
#endif
