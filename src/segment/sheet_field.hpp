// segment/sheet_field.hpp — the shared output type of sheet detectors (structure tensor,
// Hessian, OOF, phase-symmetry): per-voxel sheetness scalar + across-sheet normal.
#pragma once

#include "core/core.hpp"

#include <vector>

namespace fenix::segment {

struct SheetField {
    Volume<f32> sheetness;      // confidence in [0,1]
    std::vector<Vec3f> normal;  // size N (ZYX flat): unit across-sheet direction (sign arbitrary)
};

}  // namespace fenix::segment
