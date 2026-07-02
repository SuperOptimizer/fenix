// gui/viewer.hpp — the 4-pane viewer window (xy/xz/yz slice panes + the composite
// surface pane) with the annotation toolbar, and run_viewer() — the CLI entry that owns
// the QApplication. Crosshair-linked panes: any pane (or the surface) sets the shared
// cursor and every pane follows. Annotations save/load as versioned TOML (annotate/).
// No Q_OBJECT anywhere — all wiring is lambda connects to stock-widget signals.
#pragma once

#include "gui/panes.hpp"
#include "gui/state.hpp"
#include "io/surface.hpp"

#include <QtWidgets/QApplication>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QDoubleSpinBox>
#include <QtWidgets/QFileDialog>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QToolBar>

#include <format>
#include <memory>
#include <string>
#include <vector>

namespace fenix::gui {

class ViewerWindow : public QMainWindow {
public:
    explicit ViewerWindow(ViewerState& st) : st_(st) {
        setWindowTitle("fenix");
        resize(1400, 1000);

        auto* central = new QWidget(this);
        auto* grid = new QGridLayout(central);
        grid->setSpacing(2);
        grid->setContentsMargins(2, 2, 2, 2);
        xy_ = new SlicePane(st_, view::SliceAxis::z, central);
        xz_ = new SlicePane(st_, view::SliceAxis::y, central);
        yz_ = new SlicePane(st_, view::SliceAxis::x, central);
        surf_ = new SurfacePane(st_, central);
        grid->addWidget(xy_, 0, 0);
        grid->addWidget(xz_, 0, 1);
        grid->addWidget(yz_, 1, 0);
        grid->addWidget(surf_, 1, 1);
        setCentralWidget(central);

        build_toolbar_();

        st_.refresh_all = [this] {
            xy_->follow_cursor();
            xz_->follow_cursor();
            yz_->follow_cursor();
            surf_->mark_dirty();
        };
        // Qt<->std string bridges go through UTF-8 char* only: toStdString/fromStdString cross
        // the std::string ABI boundary, which breaks against a libstdc++-built system Qt.
        st_.status = [this](const std::string& s) {
            statusBar()->showMessage(QString::fromUtf8(s.data(), static_cast<qsizetype>(s.size())));
        };
        statusBar()->showMessage("wheel: slice · ctrl+wheel: zoom · drag: pan · right-drag: window/level · "
                                 "double-click: finish edit");
    }

private:
    void build_toolbar_() {
        QToolBar* tb = addToolBar("tools");
        tb->setMovable(false);

        auto* tool = new QComboBox(tb);
        tool->addItems({"navigate", "stroke", "radial line", "link"});
        QObject::connect(tool, &QComboBox::currentIndexChanged, [this](int i) {
            st_.finish_edit();
            st_.tool = static_cast<Tool>(i);
            st_.refresh();
        });
        tb->addWidget(tool);

        auto* kind = new QComboBox(tb);
        kind->addItems({"generic", "patch", "trace", "fiber", "kollesis", "drawing"});
        QObject::connect(kind, &QComboBox::currentIndexChanged, [this](int i) {
            st_.stroke_kind = static_cast<annotate::StrokeKind>(i);  // item order == StrokeKind order
        });
        tb->addWidget(kind);

        auto* finish = new QPushButton("finish edit", tb);
        QObject::connect(finish, &QPushButton::clicked, [this] {
            st_.finish_edit();
            st_.refresh();
        });
        tb->addWidget(finish);

        auto* wind = new QDoubleSpinBox(tb);
        wind->setRange(-10000, 10000);
        wind->setDecimals(2);
        wind->setPrefix("W=");
        tb->addWidget(wind);
        auto* label = new QPushButton("label last stroke", tb);
        QObject::connect(label, &QPushButton::clicked, [this, wind] {
            if (st_.anno.strokes.empty()) return;
            auto& s = st_.anno.strokes.back();
            s.has_winding = true;
            s.winding = static_cast<f32>(wind->value());
            st_.say(std::format("stroke '{}' labeled W={}", s.name, s.winding));
            st_.refresh();
        });
        tb->addWidget(label);

        auto* comp = new QComboBox(tb);
        comp->addItems({"mean", "max", "min", "alpha", "beer-lambert"});
        comp->setCurrentIndex(1);
        QObject::connect(comp, &QComboBox::currentIndexChanged, [this](int i) {
            st_.comp_mode = static_cast<view::CompositeMode>(i);
            surf_->mark_dirty();
        });
        tb->addWidget(comp);

        auto* save = new QPushButton("save anno", tb);
        QObject::connect(save, &QPushButton::clicked, [this] {
            st_.finish_edit();
            if (auto r = annotate::save_annotations(st_.anno, st_.anno_path); !r)
                st_.say("save failed: " + r.error().message);
            else
                st_.say(std::format("saved {} strokes, {} radials, {} links -> {}", st_.anno.strokes.size(),
                                    st_.anno.radial_lines.size(), st_.anno.links.size(), st_.anno_path));
        });
        tb->addWidget(save);

        auto* load = new QPushButton("load anno", tb);
        QObject::connect(load, &QPushButton::clicked, [this] {
            const QString f = QFileDialog::getOpenFileName(this, "load annotations", {}, "TOML (*.toml)");
            if (f.isEmpty()) return;
            const std::string path(f.toUtf8().constData());
            auto r = annotate::load_annotations(path);
            if (!r) {
                st_.say("load failed: " + r.error().message);
                return;
            }
            st_.anno = std::move(*r);
            st_.anno_path = path;
            st_.refresh();
        });
        tb->addWidget(load);
    }

    ViewerState& st_;
    SlicePane *xy_ = nullptr, *xz_ = nullptr, *yz_ = nullptr;
    SurfacePane* surf_ = nullptr;
};

// `fenix view <vol.fxvol> [--surf s.fxsurf] [--anno a.toml]`
inline Expected<int> run_viewer(std::span<const std::string_view> args, Context& /*ctx*/) {
    std::string vol_path, surf_path, anno_path;
    for (usize i = 0; i < args.size(); ++i) {
        if (args[i] == "--surf" && i + 1 < args.size()) surf_path = args[++i];
        else if (args[i] == "--anno" && i + 1 < args.size()) anno_path = args[++i];
        else if (vol_path.empty()) vol_path = args[i];
        else return err(Errc::invalid_argument, "usage: fenix view <vol.fxvol> [--surf s.fxsurf] [--anno a.toml]");
    }
    if (vol_path.empty()) return err(Errc::invalid_argument, "usage: fenix view <vol.fxvol> [--surf s.fxsurf] [--anno a.toml]");

    auto arch = codec::VolumeArchive::open(vol_path);
    if (!arch) return std::unexpected(arch.error());
    view::SliceEngine engine(*arch);

    ViewerState st;
    st.arch = &*arch;
    st.engine = &engine;
    if (!surf_path.empty()) {
        auto s = io::read_fxsurf(surf_path);
        if (!s) return std::unexpected(s.error());
        st.surface = std::move(*s);
        st.has_surface = true;
    }
    if (!anno_path.empty()) {
        if (auto a = annotate::load_annotations(anno_path)) st.anno = std::move(*a);
        st.anno_path = anno_path;  // load-if-exists; a fresh path is where saves go
    }
    const Extent3 d = arch->dims();
    st.cursor = Vec3f{static_cast<f32>(d.z) * 0.5f, static_cast<f32>(d.y) * 0.5f, static_cast<f32>(d.x) * 0.5f};

    static int argc = 1;
    static char arg0[] = "fenix";
    static char* argv[] = {arg0, nullptr};
    QApplication app(argc, argv);
    ViewerWindow w(st);
    w.show();
    return app.exec();
}

}  // namespace fenix::gui
