// gui/state.hpp — the viewer's shared model (Qt-free): the archive + engine handles, the
// loaded surface, the annotation document being edited, the linked 3D cursor, window/
// level, and the active tool. Panes read/mutate this and call refresh_all(); the single
// storage coupling point is here (taberna volume_source lesson — widgets never touch
// codec/io directly).
#pragma once

#include "annotate/annotation.hpp"
#include "codec/archive.hpp"
#include "codec/source.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "view/view.hpp"

#include <functional>
#include <string>

namespace fenix::gui {

enum class Tool : u8 { navigate, stroke, radial, link };

struct ViewerState {
    codec::VolumeSource* src = nullptr;  // local archive or streaming pyramid (io::CachedPyramid)
    view::SliceEngine* engine = nullptr;

    Surface surface;
    bool has_surface = false;

    annotate::AnnotationSet anno;
    std::string anno_path = "annotations.toml";

    Vec3f cursor{0, 0, 0};  // linked crosshair, LOD-0 voxels
    f32 win_lo = 0.0f, win_hi = 255.0f;

    int slice_step = 1;            // shift+wheel slice increment (Shift+G/H adjusts, 1..100 — VC3D scheme)
    bool show_annotations = true;  // Space toggles the overlay

    Tool tool = Tool::navigate;
    annotate::StrokeKind stroke_kind = annotate::StrokeKind::generic;
    s32 active_stroke = -1;   // stroke being drawn (index into anno.strokes)
    s32 active_radial = -1;   // radial line being drawn
    s32 link_a = -1;          // first stroke picked by the link tool

    view::CompositeMode comp_mode = view::CompositeMode::max;
    f32 comp_lo = -8.0f, comp_hi = 8.0f;

    std::function<void()> refresh_all;   // set by the window; panes call it after edits
    std::function<void()> recenter_all;  // X: every pane recenters its view on the cursor
    std::function<void(const std::string&)> status;  // status-bar line

    void refresh() const { if (refresh_all) refresh_all(); }
    void recenter() const { if (recenter_all) recenter_all(); }
    void say(const std::string& s) const { if (status) status(s); }

    // End any in-progress stroke/radial (tool switch, save, finish button).
    void finish_edit() {
        // Drop degenerate leftovers so the document never carries empty geometry.
        if (active_stroke >= 0 && anno.strokes[static_cast<usize>(active_stroke)].points.empty())
            anno.strokes.erase(anno.strokes.begin() + active_stroke);
        if (active_radial >= 0 && anno.radial_lines[static_cast<usize>(active_radial)].points.size() < 2)
            anno.radial_lines.erase(anno.radial_lines.begin() + active_radial);
        active_stroke = -1;
        active_radial = -1;
        link_a = -1;
    }
};

}  // namespace fenix::gui
