// gui/scroll3d.hpp — `fenix view-scroll`: whole-scroll multiscale rendering, rung 1 of
// the render3d plan (docs/design/render3d.md, ADR 0011): a CLIPMAP STACK of stock VTK
// volumes — a pinned coarse top level (the whole scroll, always resident: the
// never-a-hole invariant) plus nested finer boxes around the camera focus, each level
// re-gathered ASYNCHRONOUSLY as you move (double-buffered swap on a render-thread timer:
// coarse instantly, sharp seconds later — eventual consistency). Data streams per-LOD
// through CachedVolume (`<prefix>_l<k>.fxvol@<zarr-root>/<k>`), so the first visit to a
// region pulls from S3 and every later visit is local disk.
//   fenix view-scroll <cache-prefix>@<zarr-root> [levels=0,3,5] [box=288] [top=5]
//                     [surf=<fxsurf>...] [stride=4]
//   keys: [ ] { } window · d/D density · f re-center clipmap on camera focus ·
//         s surfaces · r reset · q quit    mouse: trackball
// levels: which pyramid LODs form the stack (finest first); top: the pinned context LOD.
#pragma once

#include "core/core.hpp"
#include "core/surface.hpp"
#include "io/cached_volume.hpp"
#include "io/surface.hpp"

#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageData.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkNew.h>
#include <vtkPiecewiseFunction.h>
#include <vtkPointData.h>
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
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "gui/vol3d.hpp"  // detail::surf_actor (mesh overlay), shared conventions

namespace fenix::gui {

namespace detail {

struct ClipLevel {
    s64 lod = 0;                       // pyramid level (voxel size = 2^lod LOD-0 voxels)
    s64 box = 288;                     // resident cube side, in THIS level's voxels
    std::optional<io::CachedVolume> cv;
    Extent3 ldims{};                   // this level's dims (level voxels)
    // render-side
    vtkSmartPointer<vtkVolume> volume;
    vtkNew<vtkImageData> img;
    std::vector<u8> front;             // buffer currently displayed
    Index3 front_org{-1, -1, -1};      // level-voxel origin of `front`
    // worker-side (double buffer + handoff)
    std::vector<u8> back;
    Index3 back_org{0, 0, 0};
    std::atomic<bool> ready{false};    // back buffer gathered, awaiting main-thread swap
    std::atomic<bool> busy{false};
};

struct ScrollApp {
    vtkNew<vtkRenderer> ren;
    vtkNew<vtkRenderWindow> win;
    vtkNew<vtkRenderWindowInteractor> iren;
    vtkNew<vtkTextActor> text;
    vtkNew<vtkPiecewiseFunction> opacity;
    vtkNew<vtkColorTransferFunction> color;
    std::vector<std::unique_ptr<ClipLevel>> levels;  // finest first; last = pinned top (atomics: not movable)
    std::vector<vtkSmartPointer<vtkActor>> surf_actors;
    f64 lo = 70, hi = 180, dens = 0.35;
    bool surfs_on = true;
    std::atomic<bool> quit{false};
    std::atomic<s64> want_focus[3] = {0, 0, 0};  // LOD-0 coords the worker should center on
    std::thread worker;

    void rebuild_tf() {
        opacity->RemoveAllPoints();
        opacity->AddPoint(0.0, 0.0);
        opacity->AddPoint(std::max(1.0, lo), 0.0);
        opacity->AddPoint(std::max(lo + 1.0, hi), dens);
        opacity->AddPoint(255.0, dens);
    }
    void hud() {
        char buf[300];
        std::string lv;
        for (const auto& L : levels) lv += (lv.empty() ? "" : ",") + std::to_string(L->lod);
        std::snprintf(buf,
                      sizeof buf,
                      "clipmap lods [%s]  window %.0f..%.0f  density %.2f\n"
                      "[ ] lo   { } hi   d/D density   f re-center   s surfaces   r reset   q quit",
                      lv.c_str(),
                      lo,
                      hi,
                      dens);
        text->SetInput(buf);
    }
    void refresh() {
        rebuild_tf();
        hud();
        win->Render();
    }

    bool did_first_fit = false;
    // main-thread: install a gathered back buffer into the VTK volume
    void apply_swaps() {
        bool any = false;
        for (auto& Lp : levels) {
            auto& L = *Lp;
            if (!L.ready.load(std::memory_order_acquire)) continue;
            L.front.swap(L.back);
            L.front_org = L.back_org;
            const s64 sc = s64{1} << L.lod;
            L.img->SetDimensions(static_cast<int>(L.box), static_cast<int>(L.box), static_cast<int>(L.box));
            L.img->SetSpacing(static_cast<f64>(sc), static_cast<f64>(sc), static_cast<f64>(sc));
            // world origin (LOD-0 voxels): VTK axes are (x,y,z) = reversal of our ZYX
            L.img->SetOrigin(static_cast<f64>(L.front_org.x * sc),
                             static_cast<f64>(L.front_org.y * sc),
                             static_cast<f64>(L.front_org.z * sc));
            vtkNew<vtkUnsignedCharArray> arr;
            arr->SetNumberOfComponents(1);
            arr->SetArray(L.front.data(), static_cast<vtkIdType>(L.front.size()), /*save=*/1);
            L.img->GetPointData()->SetScalars(arr);
            L.img->Modified();
            L.volume->SetVisibility(1);
            L.ready.store(false, std::memory_order_release);
            any = true;
        }
        if (any) {
            if (!did_first_fit) {
                ren->ResetCamera();
                did_first_fit = true;
            }
            win->Render();
        }
    }

    // worker thread: keep every (non-top) level's box centered on want_focus
    void worker_loop() {
        while (!quit.load(std::memory_order_relaxed)) {
            bool worked = false;
            for (usize i = 0; i + 1 < levels.size(); ++i) {  // last level pinned
                auto& L = *levels[i];
                if (L.ready.load(std::memory_order_acquire)) continue;  // swap pending
                const s64 sc = s64{1} << L.lod;
                const Index3 focus{want_focus[0].load() / sc, want_focus[1].load() / sc, want_focus[2].load() / sc};
                Index3 org{std::clamp<s64>(focus.z - L.box / 2, 0, std::max<s64>(0, L.ldims.z - L.box)),
                           std::clamp<s64>(focus.y - L.box / 2, 0, std::max<s64>(0, L.ldims.y - L.box)),
                           std::clamp<s64>(focus.x - L.box / 2, 0, std::max<s64>(0, L.ldims.x - L.box))};
                const s64 moved = std::abs(org.z - L.front_org.z) + std::abs(org.y - L.front_org.y) +
                                  std::abs(org.x - L.front_org.x);
                if (L.front_org.z >= 0 && moved < L.box / 4) continue;  // close enough
                L.busy.store(true);
                L.back.assign(static_cast<usize>(L.box * L.box * L.box), 0);
                const auto g = L.cv->gather_box_u8(org.z, org.y, org.x, L.box, L.box, L.box, L.back.data());
                L.busy.store(false);
                if (!g) continue;  // transient fetch failure: keep coarse, retry next pass
                L.back_org = org;
                L.ready.store(true, std::memory_order_release);
                worked = true;
            }
            if (!worked) std::this_thread::sleep_for(std::chrono::milliseconds(120));
        }
    }

    static void on_key(vtkObject* caller, unsigned long, void* client, void*) {
        auto* self = static_cast<ScrollApp*>(client);
        auto* it = static_cast<vtkRenderWindowInteractor*>(caller);
        const std::string_view k = it->GetKeySym() ? it->GetKeySym() : "";
        if (k == "bracketleft") self->lo = std::max(0.0, self->lo - 8);
        else if (k == "bracketright") self->lo = std::min(self->hi - 1, self->lo + 8);
        else if (k == "braceleft") self->hi = std::max(self->lo + 1, self->hi - 8);
        else if (k == "braceright") self->hi = std::min(255.0, self->hi + 8);
        else if (k == "d") self->dens = std::min(1.0, self->dens * 1.3);
        else if (k == "D") self->dens = std::max(0.02, self->dens / 1.3);
        else if (k == "f") self->recenter();
        else if (k == "s") {
            self->surfs_on = !self->surfs_on;
            for (auto& a : self->surf_actors) a->SetVisibility(self->surfs_on);
        } else if (k == "r") self->ren->ResetCamera();
        else return;
        self->refresh();
    }
    static void on_timer(vtkObject*, unsigned long, void* client, void*) {
        auto* self = static_cast<ScrollApp*>(client);
        self->apply_swaps();
    }
    void recenter() {
        f64 fp[3];
        ren->GetActiveCamera()->GetFocalPoint(fp);  // world = LOD-0 voxels, (x,y,z)
        want_focus[0].store(static_cast<s64>(fp[2]));  // z
        want_focus[1].store(static_cast<s64>(fp[1]));  // y
        want_focus[2].store(static_cast<s64>(fp[0]));  // x
    }
};

}  // namespace detail

inline Expected<int> run_view_scroll(std::span<const std::string_view> args, Context&) {
    if (args.empty())
        return err(Errc::invalid_argument,
                   "usage: view-scroll <cache-prefix>@<zarr-root> [levels=0,3,5] [box=288] "
                   "[surf=<fxsurf>...] [stride=4]");
    std::string src;
    std::vector<s64> lods{0, 3, 5};
    s64 box = 288, stride = 4;
    std::vector<std::string> surf_paths;
    for (const auto a : args) {
        auto num = [&](std::string_view key, auto& v) {
            if (!a.starts_with(key)) return false;
            const auto t = a.substr(key.size());
            std::from_chars(t.data(), t.data() + t.size(), v);
            return true;
        };
        if (num("box=", box) || num("stride=", stride)) continue;
        if (a.starts_with("levels=")) {
            lods.clear();
            usize p = 7;
            while (p < a.size()) {
                s64 v = 0;
                const auto [q, ec] = std::from_chars(a.data() + p, a.data() + a.size(), v);
                if (ec != std::errc()) break;
                lods.push_back(v);
                p = static_cast<usize>(q - a.data()) + 1;
            }
            continue;
        }
        if (a.starts_with("surf=")) {
            surf_paths.emplace_back(a.substr(5));
            continue;
        }
        src = std::string(a);
    }
    const auto at = src.find('@');
    if (at == std::string::npos)
        return err(Errc::invalid_argument, "view-scroll: source wants <cache-prefix>@<zarr-root>");
    if (lods.size() < 2) return err(Errc::invalid_argument, "view-scroll: want >=2 levels (fine..., top)");
    std::sort(lods.begin(), lods.end());  // finest first; last is the pinned top

    detail::ScrollApp app;
    for (const s64 k : lods) {
        auto L = std::make_unique<detail::ClipLevel>();
        L->lod = k;
        L->box = box;
        auto cv = io::CachedVolume::open(src.substr(0, at) + "_l" + std::to_string(k) + ".fxvol",
                                         src.substr(at + 1) + "/" + std::to_string(k));
        if (!cv) return std::unexpected(cv.error());
        L->ldims = cv->dims();
        L->cv = std::move(*cv);
        app.levels.push_back(std::move(L));
    }
    // top level: whole extent if it fits in one box, else a box'd centre (still pinned)
    {
        auto& T = *app.levels.back();
        T.box = std::min<s64>({std::max<s64>(T.ldims.z, 1), std::max<s64>(T.ldims.y, 1),
                               std::max<s64>(T.ldims.x, 1), 512});
    }

    app.rebuild_tf();
    app.color->AddRGBPoint(0.0, 0.0, 0.0, 0.0);
    app.color->AddRGBPoint(255.0, 0.93, 0.87, 0.78);
    for (auto& Lp : app.levels) {
        auto& L = *Lp;
        vtkNew<vtkSmartVolumeMapper> m;
        m->SetInputData(L.img);
        m->SetInteractiveUpdateRate(10.0);
        vtkNew<vtkVolumeProperty> prop;
        prop->SetColor(app.color);
        prop->SetScalarOpacity(app.opacity);
        prop->SetInterpolationTypeToLinear();
        prop->ShadeOn();
        prop->SetAmbient(0.25);
        prop->SetDiffuse(0.9);
        L.volume = vtkSmartPointer<vtkVolume>::New();
        L.volume->SetMapper(m);
        L.volume->SetProperty(prop);
        L.volume->SetVisibility(0);  // becomes visible at first swap
        app.ren->AddVolume(L.volume);
    }
    for (const auto& sp : surf_paths) {
        auto s = io::read_fxsurf(sp);
        if (!s) return std::unexpected(s.error());
        if (auto a = detail::surf_actor(*s, std::max<s64>(1, stride), Vec3f{0, 0, 0})) {
            app.surf_actors.push_back(a);
            app.ren->AddActor(a);
        }
    }

    app.ren->SetBackground(0.05, 0.05, 0.06);
    app.win->AddRenderer(app.ren);
    app.win->SetSize(1440, 1000);
    app.win->SetWindowName("fenix view-scroll");
    app.iren->SetRenderWindow(app.win);
    vtkNew<vtkInteractorStyleTrackballCamera> style;
    app.iren->SetInteractorStyle(style);
    app.text->GetTextProperty()->SetFontSize(15);
    app.text->GetTextProperty()->SetColor(0.75, 0.85, 0.95);
    app.text->SetPosition(12, 12);
    app.ren->AddActor2D(app.text);
    vtkNew<vtkCallbackCommand> keycb;
    keycb->SetCallback(detail::ScrollApp::on_key);
    keycb->SetClientData(&app);
    app.iren->AddObserver(vtkCommand::KeyPressEvent, keycb);

    // initial focus = volume centre; worker starts gathering immediately (top level first
    // visible since its box covers the whole extent at coarse lods)
    const auto& T = *app.levels.back();
    const s64 tsc = s64{1} << T.lod;
    app.want_focus[0].store(T.ldims.z * tsc / 2);
    app.want_focus[1].store(T.ldims.y * tsc / 2);
    app.want_focus[2].store(T.ldims.x * tsc / 2);
    // the pinned top participates once: seed it by treating it as re-gatherable this once
    app.worker = std::thread([&app] {
        // seed the top level synchronously-in-worker (whole coarse context first = the
        // never-a-hole floor), then serve the finer levels forever
        auto& Tl = *app.levels.back();
        Tl.back.assign(static_cast<usize>(Tl.box * Tl.box * Tl.box), 0);
        const Index3 torg{std::max<s64>(0, (Tl.ldims.z - Tl.box) / 2),
                          std::max<s64>(0, (Tl.ldims.y - Tl.box) / 2),
                          std::max<s64>(0, (Tl.ldims.x - Tl.box) / 2)};
        if (Tl.cv->gather_box_u8(torg.z, torg.y, torg.x, Tl.box, Tl.box, Tl.box, Tl.back.data())) {
            Tl.back_org = torg;
            Tl.ready.store(true, std::memory_order_release);
        }
        app.worker_loop();
    });

    app.iren->Initialize();
    app.iren->CreateRepeatingTimer(150);
    vtkNew<vtkCallbackCommand> timercb;
    timercb->SetCallback(detail::ScrollApp::on_timer);
    timercb->SetClientData(&app);
    app.iren->AddObserver(vtkCommand::TimerEvent, timercb);
    app.refresh();
    app.iren->Start();
    app.quit.store(true);
    if (app.worker.joinable()) app.worker.join();
    return 0;
}

}  // namespace fenix::gui

namespace {
[[maybe_unused]] const int fenix_stage_view_scroll = ::fenix::register_stage(::fenix::Stage{
    "view-scroll", "whole-scroll multiscale clipmap rendering (render3d rung 1)", ::fenix::gui::run_view_scroll});
}  // namespace
