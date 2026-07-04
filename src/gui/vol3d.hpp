// gui/vol3d.hpp — `fenix view-vol`: real-time interactive VOLUMETRIC rendering (native
// VTK). Loads an .fxvol region (or a whole archive at the LOD that fits the VRAM budget,
// via the explicit pyramid) into a GPU-raycast volume with live transfer-function keys,
// and overlays segment meshes as PBR surfaces in the SAME scene — the volumetric leg of
// the seeing-the-scroll effort (multiscale brick streaming is the planned v2; this v1 is
// one resident brick chosen by LOD).
//   fenix view-vol <fxvol> [z0 y0 x0 d [h w]] [lod=auto] [max_mb=512]
//                  [surf=<fxsurf> ...] [stride=4] [surfoff=z,y,x]
// surfoff: the fxvol's WORLD origin when it is a cropped export (eval blocks) — meshes
// carry absolute scroll coords and must be shifted into the crop frame.
//   keys: [ ] window low · { } window high · d density · D density down · s surfaces
//         on/off · r reset · q quit    mouse: trackball
#pragma once

#include "codec/archive.hpp"
#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/surface.hpp"

#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkPolyDataNormals.h>
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
#include <cmath>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace fenix::gui {

namespace detail {

struct VolApp {
    vtkNew<vtkRenderer> ren;
    vtkNew<vtkRenderWindow> win;
    vtkNew<vtkRenderWindowInteractor> iren;
    vtkNew<vtkTextActor> text;
    vtkNew<vtkPiecewiseFunction> opacity;
    f64 lo = 70, hi = 180, dens = 0.35;
    bool surfs_on = true;
    std::vector<vtkSmartPointer<vtkActor>> surf_actors;

    void rebuild_tf() {
        opacity->RemoveAllPoints();
        opacity->AddPoint(0.0, 0.0);
        opacity->AddPoint(std::max(1.0, lo), 0.0);
        opacity->AddPoint(std::max(lo + 1.0, hi), dens);
        opacity->AddPoint(255.0, dens);
    }
    void hud(s64 lod) {
        char buf[256];
        std::snprintf(buf,
                      sizeof buf,
                      "lod %lld  window %.0f..%.0f  density %.2f  surfaces %s\n"
                      "[ ] lo   { } hi   d/D density   s surfaces   r reset   q quit",
                      static_cast<long long>(lod),
                      lo,
                      hi,
                      dens,
                      surfs_on ? "on" : "off");
        text->SetInput(buf);
    }
    s64 lod_shown = 0;
    void refresh() {
        rebuild_tf();
        hud(lod_shown);
        win->Render();
    }
    static void on_key(vtkObject* caller, unsigned long, void* client, void*) {
        auto* self = static_cast<VolApp*>(client);
        auto* it = static_cast<vtkRenderWindowInteractor*>(caller);
        const std::string_view k = it->GetKeySym() ? it->GetKeySym() : "";
        if (k == "bracketleft") self->lo = std::max(0.0, self->lo - 8);
        else if (k == "bracketright") self->lo = std::min(self->hi - 1, self->lo + 8);
        else if (k == "braceleft") self->hi = std::max(self->lo + 1, self->hi - 8);
        else if (k == "braceright") self->hi = std::min(255.0, self->hi + 8);
        else if (k == "d") self->dens = std::min(1.0, self->dens * 1.3);
        else if (k == "D") self->dens = std::max(0.02, self->dens / 1.3);
        else if (k == "s") {
            self->surfs_on = !self->surfs_on;
            for (auto& a : self->surf_actors) a->SetVisibility(self->surfs_on);
        } else if (k == "r") {
            self->ren->ResetCamera();
        } else {
            return;
        }
        self->refresh();
    }
};

// strided fxsurf -> PBR actor in WORLD coords shifted into the region frame
inline vtkSmartPointer<vtkActor> surf_actor(const Surface& s, s64 stride, Vec3f org) {
    vtkNew<vtkPoints> pts;
    const s64 gu = (s.nu - 1) / stride + 1, gv = (s.nv - 1) / stride + 1;
    std::vector<vtkIdType> pid(static_cast<usize>(gu * gv), -1);
    for (s64 gj = 0; gj < gv; ++gj)
        for (s64 gi = 0; gi < gu; ++gi) {
            const s64 u = gi * stride, v = gj * stride;
            if (!s.is_valid(u, v)) continue;
            const Vec3f p = s.at(u, v) - org;
            pid[static_cast<usize>(gj * gu + gi)] = pts->InsertNextPoint(p.x, p.y, p.z);
        }
    vtkNew<vtkCellArray> tris;
    for (s64 gj = 0; gj + 1 < gv; ++gj)
        for (s64 gi = 0; gi + 1 < gu; ++gi) {
            const vtkIdType a = pid[static_cast<usize>(gj * gu + gi)];
            const vtkIdType b = pid[static_cast<usize>(gj * gu + gi + 1)];
            const vtkIdType c = pid[static_cast<usize>((gj + 1) * gu + gi)];
            const vtkIdType d = pid[static_cast<usize>((gj + 1) * gu + gi + 1)];
            if (a < 0 || b < 0 || c < 0 || d < 0) continue;
            const vtkIdType t1[3] = {a, b, d}, t2[3] = {a, d, c};
            tris->InsertNextCell(3, t1);
            tris->InsertNextCell(3, t2);
        }
    if (tris->GetNumberOfCells() == 0) return nullptr;
    vtkNew<vtkPolyData> poly;
    poly->SetPoints(pts);
    poly->SetPolys(tris);
    vtkNew<vtkPolyDataNormals> nrm;
    nrm->SetInputData(poly);
    nrm->SplittingOff();
    vtkNew<vtkPolyDataMapper> m;
    m->SetInputConnection(nrm->GetOutputPort());
    auto a = vtkSmartPointer<vtkActor>::New();
    a->SetMapper(m);
    auto* pr = a->GetProperty();
    pr->SetInterpolationToPBR();
    pr->SetRoughness(0.6);
    pr->SetMetallic(0.0);
    pr->SetColor(0.9, 0.35, 0.2);
    return a;
}

}  // namespace detail

inline Expected<int> run_view_vol(std::span<const std::string_view> args, Context&) {
    if (args.empty())
        return err(Errc::invalid_argument,
                   "usage: view-vol <fxvol> [z0 y0 x0 d [h w]] [lod=auto] [max_mb=512] [surf=<fxsurf>...] [stride=4]");
    s64 lod = -1, max_mb = 512, stride = 4;
    Vec3f surfoff{0, 0, 0};
    std::vector<s64> box;
    std::vector<std::string> surf_paths;
    std::string vol_path;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("lod=", lod) || num("max_mb=", max_mb) || num("stride=", stride)) continue;
        if (a.starts_with("surf=")) {
            surf_paths.emplace_back(a.substr(5));
            continue;
        }
        if (a.starts_with("surfoff=")) {
            const auto t = a.substr(8);
            const auto c1 = t.find(','), c2 = t.rfind(',');
            if (c1 != std::string_view::npos && c2 != c1) {
                f64 z = 0, y = 0, x = 0;
                std::from_chars(t.data(), t.data() + c1, z);
                std::from_chars(t.data() + c1 + 1, t.data() + c2, y);
                std::from_chars(t.data() + c2 + 1, t.data() + t.size(), x);
                surfoff = Vec3f{static_cast<f32>(z), static_cast<f32>(y), static_cast<f32>(x)};
            }
            continue;
        }
        s64 v = 0;
        const auto [p, ec] = std::from_chars(a.data(), a.data() + a.size(), v);
        if (ec == std::errc() && p == a.data() + a.size()) {
            box.push_back(v);
            continue;
        }
        vol_path = std::string(a);
    }
    if (vol_path.empty()) return err(Errc::invalid_argument, "view-vol: no fxvol");
    if (!box.empty() && box.size() != 4 && box.size() != 6)
        return err(Errc::invalid_argument, "view-vol: region wants z0 y0 x0 d  or  z0 y0 x0 d h w");

    auto arch = codec::VolumeArchive::open(vol_path);
    if (!arch) return std::unexpected(arch.error());
    arch->reserve_cache(u64{2} << 30);
    const Extent3 full = arch->dims();
    Index3 org{0, 0, 0};
    Extent3 ext = full;
    if (!box.empty()) {
        org = Index3{box[0], box[1], box[2]};
        ext = box.size() == 6 ? Extent3{box[3], box[4], box[5]} : Extent3{box[3], box[3], box[3]};
    }
    // LOD pick: smallest lod whose region fits max_mb (u8 voxels)
    const s64 nlods = static_cast<s64>(arch->nlods());
    if (lod < 0) {
        lod = 0;
        while (lod + 1 < nlods) {
            const f64 mb = static_cast<f64>(ext.z >> lod) * static_cast<f64>(ext.y >> lod) *
                           static_cast<f64>(ext.x >> lod) / (1024.0 * 1024.0);
            if (mb <= static_cast<f64>(max_mb)) break;
            ++lod;
        }
    }
    lod = std::clamp<s64>(lod, 0, std::max<s64>(0, nlods - 1));
    const s64 sc = s64{1} << lod;
    const Extent3 le{std::max<s64>(1, ext.z / sc), std::max<s64>(1, ext.y / sc), std::max<s64>(1, ext.x / sc)};
    const f64 mb = static_cast<f64>(le.z) * static_cast<f64>(le.y) * static_cast<f64>(le.x) / (1024.0 * 1024.0);
    if (mb > 4096.0) return err(Errc::invalid_argument, "view-vol: region too large even at max lod — give a box");
    log(LogLevel::info, "view-vol: {}x{}x{} at lod {} ({:.0f} MB)", le.z, le.y, le.x, lod, mb);

    // gather (u8) — one resident brick for v1
    std::vector<u8> buf(static_cast<usize>(le.z * le.y * le.x));
    if (auto g = arch->gather_box_u8(lod, org.z / sc, org.y / sc, org.x / sc, le.z, le.y, le.x, buf.data()); !g)
        return std::unexpected(g.error());

    detail::VolApp app;
    vtkNew<vtkImageData> img;
    img->SetDimensions(static_cast<int>(le.x), static_cast<int>(le.y), static_cast<int>(le.z));
    img->SetSpacing(static_cast<f64>(sc), static_cast<f64>(sc), static_cast<f64>(sc));
    vtkNew<vtkUnsignedCharArray> arr;
    arr->SetNumberOfComponents(1);
    arr->SetArray(buf.data(), static_cast<vtkIdType>(buf.size()), /*save=*/1);
    img->GetPointData()->SetScalars(arr);

    vtkNew<vtkSmartVolumeMapper> vmap;
    vmap->SetInputData(img);
    vmap->SetInteractiveUpdateRate(10.0);  // stay interactive: coarser sampling while moving
    vtkNew<vtkColorTransferFunction> color;
    color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
    color->AddRGBPoint(255.0, 0.93, 0.87, 0.78);  // papyrus-warm ramp
    app.rebuild_tf();
    vtkNew<vtkVolumeProperty> prop;
    prop->SetColor(color);
    prop->SetScalarOpacity(app.opacity);
    prop->SetInterpolationTypeToLinear();
    prop->ShadeOn();  // gradient shading: the volumetric "PBR-ish" look
    prop->SetAmbient(0.25);
    prop->SetDiffuse(0.9);
    prop->SetSpecular(0.2);
    vtkNew<vtkVolume> vol;
    vol->SetMapper(vmap);
    vol->SetProperty(prop);
    app.ren->AddVolume(vol);

    for (const auto& sp : surf_paths) {
        auto s = io::read_fxsurf(sp);
        if (!s) return std::unexpected(s.error());
        // mesh world coords -> region-local, note vtkImageData spacing keeps lod scale
        if (auto a = detail::surf_actor(
                *s,
                std::max<s64>(1, stride),
                surfoff + Vec3f{static_cast<f32>(org.z), static_cast<f32>(org.y), static_cast<f32>(org.x)})) {
            // Surface coords are ZYX; the actor builder wrote (x,y,z) points — consistent
            // with the image (x,y,z) axes above.
            app.surf_actors.push_back(a);
            app.ren->AddActor(a);
        }
    }

    app.lod_shown = lod;
    app.ren->SetBackground(0.05, 0.05, 0.06);
    app.win->AddRenderer(app.ren);
    app.win->SetSize(1440, 1000);
    app.win->SetWindowName("fenix view-vol");
    app.iren->SetRenderWindow(app.win);
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    app.iren->SetInteractorStyle(style);
    app.text->GetTextProperty()->SetFontSize(15);
    app.text->GetTextProperty()->SetColor(0.75, 0.85, 0.95);
    app.text->SetPosition(12, 12);
    app.ren->AddActor2D(app.text);
    vtkNew<vtkCallbackCommand> keycb;
    keycb->SetCallback(detail::VolApp::on_key);
    keycb->SetClientData(&app);
    app.iren->AddObserver(vtkCommand::KeyPressEvent, keycb);
    app.iren->Initialize();
    app.ren->ResetCamera();
    app.refresh();
    app.iren->Start();
    return 0;
}

}  // namespace fenix::gui

namespace {
[[maybe_unused]] const int fenix_stage_view_vol = ::fenix::register_stage(::fenix::Stage{
    "view-vol", "real-time interactive volumetric rendering (+PBR segment overlay)", ::fenix::gui::run_view_vol});
}  // namespace
