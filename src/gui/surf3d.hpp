// gui/surf3d.hpp — `fenix view-surf`: real-time PBR segment rendering (native VTK).
// The segment's uv grid becomes a triangle mesh; the CT-baked papyrus texture
// (fenix surf-bake) drapes over it as the PBR base color; a movable RAKING key light
// brings out fiber relief the way papyrologists rake real fragments. This is the
// "make the segments look good" renderer — the seeing-the-scroll effort's first face.
//   fenix view-surf <fxsurf> [tex=baked.jpg] [stride=2] [roughness=0.65]
//   keys: arrows move the key light (raking!) · +/- light intensity · t texture on/off ·
//         g wireframe · r reset camera · q quit    mouse: trackball rotate/zoom/pan
#pragma once

#include "core/core.hpp"
#include "io/surface.hpp"

#include <vtkActor.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkCellArray.h>
#include <vtkFloatArray.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkJPEGReader.h>
#include <vtkLight.h>
#include <vtkNew.h>
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
#include <vtkTextActor.h>
#include <vtkTextProperty.h>
#include <vtkTexture.h>

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

struct SurfApp {
    vtkNew<vtkRenderer> ren;
    vtkNew<vtkRenderWindow> win;
    vtkNew<vtkRenderWindowInteractor> iren;
    vtkNew<vtkLight> key;
    vtkNew<vtkTextActor> text;
    vtkSmartPointer<vtkActor> actor;
    f64 az = 40, el = 35, inten = 1.2;
    bool tex_on = true, wire = false;
    f64 center[3] = {0, 0, 0}, radius = 1;

    void place_light() {
        const f64 a = az * 3.14159265 / 180.0, e = el * 3.14159265 / 180.0;
        const f64 d = radius * 3.0;
        key->SetPosition(center[0] + d * std::cos(e) * std::cos(a),
                         center[1] + d * std::cos(e) * std::sin(a),
                         center[2] + d * std::sin(e));
        key->SetFocalPoint(center[0], center[1], center[2]);
        key->SetIntensity(inten);
    }
    void hud() {
        char buf[256];
        std::snprintf(buf,
                      sizeof buf,
                      "light az %.0f  el %.0f  intensity %.1f  texture %s\n"
                      "arrows rake light   +/- intensity   t texture   g wireframe   r reset   q quit",
                      az,
                      el,
                      inten,
                      tex_on ? "on" : "off");
        text->SetInput(buf);
    }
    void refresh() {
        place_light();
        hud();
        win->Render();
    }
    static void on_key(vtkObject* caller, unsigned long, void* client, void*) {
        auto* self = static_cast<SurfApp*>(client);
        auto* it = static_cast<vtkRenderWindowInteractor*>(caller);
        const std::string_view k = it->GetKeySym() ? it->GetKeySym() : "";
        if (k == "Left") self->az -= 10;
        else if (k == "Right") self->az += 10;
        else if (k == "Up") self->el = std::min(85.0, self->el + 8);
        else if (k == "Down") self->el = std::max(2.0, self->el - 8);
        else if (k == "plus" || k == "equal") self->inten = std::min(4.0, self->inten * 1.25);
        else if (k == "minus") self->inten = std::max(0.1, self->inten / 1.25);
        else if (k == "t") {
            self->tex_on = !self->tex_on;
            self->actor->GetProperty()->SetInterpolationToPBR();  // keep PBR; drop/re-add texture
            if (self->tex_on && self->actor->GetTexture()) self->actor->GetProperty()->SetColor(1, 1, 1);
            self->actor->SetVisibility(1);
            if (auto* t = self->actor->GetTexture()) t->SetBlendingMode(
                self->tex_on ? vtkTexture::VTK_TEXTURE_BLENDING_MODE_MODULATE
                             : vtkTexture::VTK_TEXTURE_BLENDING_MODE_NONE);
        } else if (k == "g") {
            self->wire = !self->wire;
            if (self->wire) self->actor->GetProperty()->SetRepresentationToWireframe();
            else self->actor->GetProperty()->SetRepresentationToSurface();
        } else if (k == "r") {
            self->ren->ResetCamera();
        } else {
            return;
        }
        self->refresh();
    }
};

}  // namespace detail

inline Expected<int> run_view_surf(std::span<const std::string_view> args, Context&) {
    if (args.empty())
        return err(Errc::invalid_argument,
                   "usage: view-surf <fxsurf> [tex=baked.jpg] [stride=2] [roughness=0.65]");
    s64 stride = 2;
    f64 roughness = 0.65;
    std::string tex_path, surf_path;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("stride=", stride) || num("roughness=", roughness)) continue;
        if (a.starts_with("tex=")) {
            tex_path = std::string(a.substr(4));
            continue;
        }
        surf_path = std::string(a);
    }
    if (surf_path.empty()) return err(Errc::invalid_argument, "view-surf: no fxsurf");
    stride = std::clamp<s64>(stride, 1, 32);
    auto s = io::read_fxsurf(surf_path);
    if (!s) return std::unexpected(s.error());

    // uv grid -> triangles at `stride` (only cells whose 4 strided corners are valid);
    // tcoords address the FULL-RES baked texture so detail survives geometric striding
    vtkNew<vtkPoints> pts;
    vtkNew<vtkFloatArray> tc;
    tc->SetNumberOfComponents(2);
    tc->SetName("uv");
    const s64 gu = (s->nu - 1) / stride + 1, gv = (s->nv - 1) / stride + 1;
    std::vector<vtkIdType> pid(static_cast<usize>(gu * gv), -1);
    for (s64 gj = 0; gj < gv; ++gj)
        for (s64 gi = 0; gi < gu; ++gi) {
            const s64 u = gi * stride, v = gj * stride;
            if (!s->is_valid(u, v)) continue;
            const Vec3f p = s->at(u, v);
            pid[static_cast<usize>(gj * gu + gi)] = pts->InsertNextPoint(p.x, p.y, p.z);  // world XYZ
            tc->InsertNextTuple2(static_cast<f64>(u) / static_cast<f64>(std::max<s64>(1, s->nu - 1)),
                                 static_cast<f64>(v) / static_cast<f64>(std::max<s64>(1, s->nv - 1)));
        }
    vtkNew<vtkCellArray> tris;
    for (s64 gj = 0; gj + 1 < gv; ++gj)
        for (s64 gi = 0; gi + 1 < gu; ++gi) {
            const vtkIdType a2 = pid[static_cast<usize>(gj * gu + gi)];
            const vtkIdType b2 = pid[static_cast<usize>(gj * gu + gi + 1)];
            const vtkIdType c2 = pid[static_cast<usize>((gj + 1) * gu + gi)];
            const vtkIdType d2 = pid[static_cast<usize>((gj + 1) * gu + gi + 1)];
            if (a2 < 0 || b2 < 0 || c2 < 0 || d2 < 0) continue;
            const vtkIdType t1[3] = {a2, b2, d2}, t2[3] = {a2, d2, c2};
            tris->InsertNextCell(3, t1);
            tris->InsertNextCell(3, t2);
        }
    if (tris->GetNumberOfCells() == 0) return err(Errc::invalid_argument, "view-surf: no valid cells at this stride");
    vtkNew<vtkPolyData> poly;
    poly->SetPoints(pts);
    poly->SetPolys(tris);
    poly->GetPointData()->SetTCoords(tc);

    vtkNew<vtkPolyDataNormals> normals;
    normals->SetInputData(poly);
    normals->SplittingOff();
    normals->ConsistencyOn();
    vtkNew<vtkPolyDataMapper> mapper;
    mapper->SetInputConnection(normals->GetOutputPort());

    detail::SurfApp app;
    app.actor = vtkSmartPointer<vtkActor>::New();
    app.actor->SetMapper(mapper);
    auto* prop = app.actor->GetProperty();
    prop->SetInterpolationToPBR();
    prop->SetRoughness(roughness);
    prop->SetMetallic(0.0);
    prop->SetColor(0.87, 0.77, 0.62);  // papyrus tone under an untextured render
    if (!tex_path.empty()) {
        vtkNew<vtkJPEGReader> jr;
        jr->SetFileName(tex_path.c_str());
        vtkNew<vtkTexture> tex;
        tex->SetInputConnection(jr->GetOutputPort());
        tex->InterpolateOn();
        tex->MipmapOn();
        app.actor->SetTexture(tex);
        prop->SetColor(1, 1, 1);
    }
    app.ren->AddActor(app.actor);
    app.ren->SetBackground(0.05, 0.05, 0.06);

    app.win->AddRenderer(app.ren);
    app.win->SetSize(1440, 1000);
    app.win->SetWindowName("fenix view-surf");
    app.iren->SetRenderWindow(app.win);
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    app.iren->SetInteractorStyle(style);

    // lighting: one strong movable KEY (the raking light) + a dim fill so shadows read
    app.ren->RemoveAllLights();
    app.key->SetLightTypeToSceneLight();
    app.key->SetColor(1.0, 0.97, 0.9);
    app.ren->AddLight(app.key);
    vtkNew<vtkLight> fill;
    fill->SetLightTypeToHeadlight();
    fill->SetIntensity(0.25);
    fill->SetColor(0.8, 0.85, 1.0);
    app.ren->AddLight(fill);

    app.text->GetTextProperty()->SetFontSize(15);
    app.text->GetTextProperty()->SetColor(0.75, 0.85, 0.95);
    app.text->SetPosition(12, 12);
    app.ren->AddActor2D(app.text);

    vtkNew<vtkCallbackCommand> keycb;
    keycb->SetCallback(detail::SurfApp::on_key);
    keycb->SetClientData(&app);
    app.iren->AddObserver(vtkCommand::KeyPressEvent, keycb);

    app.iren->Initialize();
    app.ren->ResetCamera();
    {
        f64 b[6];
        app.actor->GetBounds(b);
        app.center[0] = (b[0] + b[1]) / 2;
        app.center[1] = (b[2] + b[3]) / 2;
        app.center[2] = (b[4] + b[5]) / 2;
        app.radius = std::max({b[1] - b[0], b[3] - b[2], b[5] - b[4]}) / 2 + 1;
    }
    app.refresh();
    log(LogLevel::info,
        "view-surf: {} pts, {} tris, stride {}, tex {}",
        pts->GetNumberOfPoints(),
        tris->GetNumberOfCells(),
        stride,
        tex_path.empty() ? "(none)" : tex_path.c_str());
    app.iren->Start();
    return 0;
}

}  // namespace fenix::gui

namespace {
[[maybe_unused]] const int fenix_stage_view_surf = ::fenix::register_stage(::fenix::Stage{
    "view-surf", "real-time PBR segment renderer (CT-baked papyrus texture, raking light)", ::fenix::gui::run_view_surf});
}  // namespace
