// gui/chunk3d.hpp — `fenix view-chunk`: NATIVE VTK 3D chunk-triage viewer. Renders a
// .qcchunk (fenix qc-chunk: CT u8 + rasterized band u8) as a GPU-raycast volume with the
// mesh band as a red isosurface — rotate freely where 2D slices fail (oblique sheets,
// tangled wraps). Pure VTK, no Qt widgets (this is the "later GPU volume-render pane"
// the gui CLAUDE reserved VTK for). Lives in the GUI firewall TU (apps/gui.cpp).
//   fenix view-chunk [umb=y,x] <chunk.qcchunk...>
//   keys: n/p next/prev chunk · b band(red) · o others(blue) · f faces (recto=green /
//         verso=cyan, needs umb=) · [ ] CT window low · { } CT window high · r reset · q quit
// umb=y,x enables FACE LABELS (taberna air_trace lineage, rewritten): per material voxel,
// air within 3 steps along the umbilicus-OUTWARD yx radial = RECTO face (green), air
// inward = VERSO (cyan). Air = below the chunk Otsu threshold. The radial is taken
// constant per chunk (the umbilicus is thousands of voxels away).
#pragma once

#include "core/core.hpp"
#include "preprocess/aircut.hpp"

#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkFlyingEdges3D.h>
#include <vtkImageData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkSmartVolumeMapper.h>
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkUnsignedCharArray.h>
#include <vtkVolume.h>
#include <vtkVolumeProperty.h>

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::gui {

namespace detail {

struct QcChunk {
    s64 size = 0;
    std::string name, header;
    s64 org[3] = {0, 0, 0};  // chunk origin in volume coords (ZYX)
    std::vector<u8> ct, band;
    std::vector<u8> primary, others;  // band split: ==255 (this mesh) / ==128 (other segments)
    std::vector<u8> recto, verso;     // face labels (computed when umb= given)
};

// taberna air_trace, rewritten: material voxel with air within `k` steps along the
// umbilicus-outward yx radial -> RECTO face; air inward -> VERSO. out_y/out_x = the
// chunk-constant outward unit vector.
inline void compute_faces(QcChunk& c, f64 out_y, f64 out_x) {
    const s64 n = c.size;
    const usize tensor = c.ct.size();
    const u8 thr = preprocess::otsu_threshold_u8(std::span<const u8>(c.ct.data(), tensor));
    c.recto.assign(tensor, 0);
    c.verso.assign(tensor, 0);
    auto at = [&](s64 z, s64 y, s64 x) { return c.ct[static_cast<usize>((z * n + y) * n + x)]; };
    for (s64 z = 0; z < n; ++z)
        for (s64 y = 0; y < n; ++y)
            for (s64 x = 0; x < n; ++x) {
                if (at(z, y, x) < thr) continue;  // air
                bool outair = false, inair = false;
                for (int k = 1; k <= 3; ++k) {
                    const s64 oy = y + static_cast<s64>(std::lround(out_y * k));
                    const s64 ox = x + static_cast<s64>(std::lround(out_x * k));
                    const s64 iy = y - static_cast<s64>(std::lround(out_y * k));
                    const s64 ix = x - static_cast<s64>(std::lround(out_x * k));
                    if (oy >= 0 && oy < n && ox >= 0 && ox < n && at(z, oy, ox) < thr) outair = true;
                    if (iy >= 0 && iy < n && ix >= 0 && ix < n && at(z, iy, ix) < thr) inair = true;
                }
                const usize i = static_cast<usize>((z * n + y) * n + x);
                if (outair) c.recto[i] = 255;
                else if (inair) c.verso[i] = 255;
            }
}

inline Expected<QcChunk> read_qcchunk(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return err(Errc::not_found, "view-chunk: cannot open " + path);
    QcChunk c;
    char hdr[512] = {};
    if (!std::fgets(hdr, sizeof hdr, f)) {
        std::fclose(f);
        return err(Errc::invalid_argument, "view-chunk: no header in " + path);
    }
    c.header = hdr;
    long long n = 0;
    if (const char* p = std::strstr(hdr, "\"size\":")) std::sscanf(p, "\"size\":%lld", &n);
    if (n < 8 || n > 512) {
        std::fclose(f);
        return err(Errc::invalid_argument, "view-chunk: bad size in " + path);
    }
    c.size = n;
    const usize tensor = static_cast<usize>(n) * static_cast<usize>(n) * static_cast<usize>(n);
    c.ct.resize(tensor);
    c.band.resize(tensor);
    if (std::fread(c.ct.data(), 1, tensor, f) != tensor || std::fread(c.band.data(), 1, tensor, f) != tensor) {
        std::fclose(f);
        return err(Errc::invalid_argument, "view-chunk: truncated payload in " + path);
    }
    std::fclose(f);
    {
        long long oz = 0, oy = 0, ox = 0;
        if (const char* q = std::strstr(hdr, "\"origin\":[")) std::sscanf(q, "\"origin\":[%lld,%lld,%lld]", &oz, &oy, &ox);
        c.org[0] = oz;
        c.org[1] = oy;
        c.org[2] = ox;
    }
    c.primary.resize(tensor);
    c.others.resize(tensor);
    for (usize i = 0; i < tensor; ++i) {
        c.primary[i] = c.band[i] == 255 ? 255 : 0;
        c.others[i] = c.band[i] == 128 ? 255 : 0;
    }
    c.name = path;
    if (const auto sl = c.name.rfind('/'); sl != std::string::npos) c.name = c.name.substr(sl + 1);
    return c;
}

// vtkImageData wrapping a u8 buffer WITHOUT copying (save=1: VTK must not free it; the
// QcChunk vector stays alive in ChunkApp for the app's lifetime).
inline vtkSmartPointer<vtkImageData> wrap_u8(std::vector<u8>& buf, s64 n) {
    auto img = vtkSmartPointer<vtkImageData>::New();
    img->SetDimensions(static_cast<int>(n), static_cast<int>(n), static_cast<int>(n));
    vtkNew<vtkUnsignedCharArray> arr;
    arr->SetNumberOfComponents(1);
    arr->SetArray(buf.data(), static_cast<vtkIdType>(buf.size()), /*save=*/1);
    img->GetPointData()->SetScalars(arr);
    return img;
}

struct ChunkApp {
    std::vector<QcChunk> chunks;
    usize cur = 0;
    f64 lo = 60, hi = 160;
    bool band_on = true;

    vtkNew<vtkRenderer> ren;
    vtkNew<vtkRenderWindow> win;
    vtkNew<vtkRenderWindowInteractor> iren;
    vtkNew<vtkTextActor> text;
    vtkNew<vtkPiecewiseFunction> opacity;
    vtkSmartPointer<vtkVolume> volume;
    vtkSmartPointer<vtkActor> band_actor;
    vtkSmartPointer<vtkActor> others_actor;
    vtkSmartPointer<vtkActor> recto_actor, verso_actor;
    bool others_on = true;
    bool faces_on = false;
    f64 umb_y = 1e30, umb_x = 1e30;

    void rebuild_opacity() {
        // CT: fully transparent below the window, linear ramp to 50% at the top (never
        // more — the surface must stay readable through the volume); band: opaque.
        opacity->RemoveAllPoints();
        opacity->AddPoint(0.0, 0.0);
        opacity->AddPoint(std::max(1.0, lo), 0.0);
        opacity->AddPoint(std::max(lo + 1.0, hi), 0.5);
        opacity->AddPoint(255.0, 0.5);
    }

    void show(usize i) {
        cur = i % chunks.size();
        QcChunk& c = chunks[cur];
        if (volume) ren->RemoveVolume(volume);
        if (band_actor) ren->RemoveActor(band_actor);

        auto ct_img = wrap_u8(c.ct, c.size);
        vtkNew<vtkSmartVolumeMapper> vmap;
        vmap->SetInputData(ct_img);
        vtkNew<vtkColorTransferFunction> color;
        color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
        color->AddRGBPoint(255.0, 0.95, 0.95, 0.95);
        rebuild_opacity();
        vtkNew<vtkVolumeProperty> prop;
        prop->SetColor(color);
        prop->SetScalarOpacity(opacity);
        prop->SetInterpolationTypeToLinear();
        prop->ShadeOff();
        volume = vtkSmartPointer<vtkVolume>::New();
        volume->SetMapper(vmap);
        volume->SetProperty(prop);
        ren->AddVolume(volume);

        if (others_actor) ren->RemoveActor(others_actor);
        auto make_iso = [&](std::vector<u8>& mask, f64 r, f64 g, f64 b2) {
            auto img = wrap_u8(mask, c.size);
            vtkNew<vtkFlyingEdges3D> iso;
            iso->SetInputData(img);
            iso->SetValue(0, 128.0);
            iso->ComputeNormalsOn();
            vtkNew<vtkPolyDataMapper> m;
            m->SetInputConnection(iso->GetOutputPort());
            m->ScalarVisibilityOff();
            auto a = vtkSmartPointer<vtkActor>::New();
            a->SetMapper(m);
            a->GetProperty()->SetColor(r, g, b2);
            a->GetProperty()->SetOpacity(1.0);
            return a;
        };
        band_actor = make_iso(c.primary, 1.0, 0.15, 0.1);   // THIS mesh: red
        band_actor->SetVisibility(band_on);
        ren->AddActor(band_actor);
        others_actor = make_iso(c.others, 0.15, 0.5, 1.0);  // other segments: blue
        others_actor->SetVisibility(others_on);
        ren->AddActor(others_actor);
        if (recto_actor) ren->RemoveActor(recto_actor);
        if (verso_actor) ren->RemoveActor(verso_actor);
        recto_actor = nullptr;
        verso_actor = nullptr;
        if (umb_y < 1e29) {
            if (c.recto.empty()) {
                f64 ry = static_cast<f64>(c.org[1] + c.size / 2) - umb_y;
                f64 rx = static_cast<f64>(c.org[2] + c.size / 2) - umb_x;
                const f64 rn = std::sqrt(ry * ry + rx * rx);
                if (rn > 1.0) compute_faces(c, ry / rn, rx / rn);
            }
            if (!c.recto.empty()) {
                recto_actor = make_iso(c.recto, 0.15, 0.95, 0.25);  // recto: green
                verso_actor = make_iso(c.verso, 0.1, 0.85, 0.9);    // verso: cyan
                recto_actor->SetVisibility(faces_on);
                verso_actor->SetVisibility(faces_on);
                ren->AddActor(recto_actor);
                ren->AddActor(verso_actor);
            }
        }

        update_text();
        ren->ResetCamera();
        win->Render();
    }

    void update_text() {
        const QcChunk& c = chunks[cur];
        char buf[512];
        std::snprintf(buf,
                      sizeof buf,
                      "%s  (%zu/%zu)  window %.0f..%.0f  band %s  others %s  faces %s\n"
                      "n/p chunk   b band(red)   o others(blue)   f faces(grn/cyn)   [ ] lo   { } hi   r reset   q quit",
                      c.name.c_str(),
                      cur + 1,
                      chunks.size(),
                      lo,
                      hi,
                      band_on ? "on" : "off",
                      others_on ? "on" : "off",
                      faces_on ? "on" : "off");
        text->SetInput(buf);
    }

    void refresh() {
        rebuild_opacity();
        update_text();
        win->Render();
    }

    static void on_key(vtkObject* caller, unsigned long, void* client, void*) {
        auto* self = static_cast<ChunkApp*>(client);
        auto* it = static_cast<vtkRenderWindowInteractor*>(caller);
        const std::string_view k = it->GetKeySym() ? it->GetKeySym() : "";
        if (k == "n") self->show(self->cur + 1);
        else if (k == "p") self->show(self->cur + self->chunks.size() - 1);
        else if (k == "b") {
            self->band_on = !self->band_on;
            if (self->band_actor) self->band_actor->SetVisibility(self->band_on);
            self->refresh();
        } else if (k == "f") {
            self->faces_on = !self->faces_on;
            if (self->recto_actor) self->recto_actor->SetVisibility(self->faces_on);
            if (self->verso_actor) self->verso_actor->SetVisibility(self->faces_on);
            self->refresh();
        } else if (k == "o") {
            self->others_on = !self->others_on;
            if (self->others_actor) self->others_actor->SetVisibility(self->others_on);
            self->refresh();
        } else if (k == "bracketleft") { self->lo = std::max(0.0, self->lo - 8); self->refresh(); }
        else if (k == "bracketright") { self->lo = std::min(self->hi - 1, self->lo + 8); self->refresh(); }
        else if (k == "braceleft") { self->hi = std::max(self->lo + 1, self->hi - 8); self->refresh(); }
        else if (k == "braceright") { self->hi = std::min(255.0, self->hi + 8); self->refresh(); }
        else if (k == "r") {
            self->ren->ResetCamera();
            self->win->Render();
        }
    }
};

}  // namespace detail

inline Expected<int> run_view_chunk(std::span<const std::string_view> args, Context&) {
    if (args.empty()) return err(Errc::invalid_argument, "usage: view-chunk [umb=y,x] <chunk.qcchunk...>");
    detail::ChunkApp app;
    for (const auto a : args) {
        if (a.starts_with("umb=")) {
            const auto t = a.substr(4);
            const auto comma = t.find(',');
            if (comma != std::string_view::npos) {
                std::from_chars(t.data() + 4 - 4, t.data() + comma, app.umb_y);
                std::from_chars(t.data() + comma + 1, t.data() + t.size(), app.umb_x);
            }
            app.faces_on = true;
            continue;
        }
        auto c = detail::read_qcchunk(std::string(a));
        if (!c) return std::unexpected(c.error());
        app.chunks.push_back(std::move(*c));
    }
    if (app.chunks.empty()) return err(Errc::invalid_argument, "view-chunk: no chunks");
    app.ren->SetBackground(0.04, 0.04, 0.05);
    app.win->AddRenderer(app.ren);
    app.win->SetSize(1280, 960);
    app.win->SetWindowName("fenix view-chunk");
    app.iren->SetRenderWindow(app.win);
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    app.iren->SetInteractorStyle(style);

    app.text->GetTextProperty()->SetFontSize(15);
    app.text->GetTextProperty()->SetColor(0.75, 0.85, 0.95);
    app.text->SetPosition(12, 12);
    app.ren->AddActor2D(app.text);

    vtkNew<vtkCallbackCommand> keycb;
    keycb->SetCallback(detail::ChunkApp::on_key);
    keycb->SetClientData(&app);
    app.iren->AddObserver(vtkCommand::KeyPressEvent, keycb);

    app.iren->Initialize();
    app.show(0);
    app.iren->Start();
    return 0;
}

}  // namespace fenix::gui

namespace {
[[maybe_unused]] const int fenix_stage_view_chunk = ::fenix::register_stage(::fenix::Stage{
    "view-chunk", "native VTK 3D chunk-triage viewer (CT volume + band isosurface)", ::fenix::gui::run_view_chunk});
}  // namespace
