// gui/panes.hpp — the viewer widgets. SlicePane: one axis-aligned pane (xy/xz/yz) —
// engine-rendered slice blit + crosshair + annotation overlay + the annotation tools
// (co-winding strokes, ridge-snapped radial winding lines, links). SurfacePane: the
// composite surface view. Plain QWidget subclasses, deliberately NO Q_OBJECT (no moc —
// the GUI stays header-only); all signalling is via ViewerState callbacks.
#pragma once

#include "gui/state.hpp"

#include <QtGui/QCursor>
#include <QtGui/QGuiApplication>
#include <QtGui/QImage>
#include <QtGui/QKeyEvent>
#include <QtGui/QMouseEvent>
#include <QtGui/QPainter>
#include <QtGui/QPen>
#include <QtGui/QWheelEvent>
#include <QtWidgets/QWidget>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>

namespace fenix::gui {

namespace detail {
inline QColor stroke_color(annotate::StrokeKind k) {
    switch (k) {
        case annotate::StrokeKind::kollesis: return {255, 140, 0};
        case annotate::StrokeKind::fiber: return {0, 220, 120};
        case annotate::StrokeKind::patch_extract: return {80, 160, 255};
        case annotate::StrokeKind::trace_extract: return {160, 120, 255};
        case annotate::StrokeKind::drawing: return {255, 100, 160};
        default: return {255, 230, 60};
    }
}
}  // namespace detail

class SlicePane : public QWidget {
public:
    SlicePane(ViewerState& st, view::SliceAxis axis, QWidget* parent = nullptr)
        : QWidget(parent), st_(st), axis_(axis) {
        setMinimumSize(160, 160);
        setMouseTracking(true);
        // Focus follows the mouse so per-pane keys (arrows, zoom, M/R) act on the hovered pane,
        // matching VC3D's active-viewer resolution.
        setFocusPolicy(Qt::ClickFocus);
        const Extent3 d = st_.arch->dims();
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        slice_ = static_cast<f32>(view::detail::axis_of(d, an)) * 0.5f;
        center_u_ = static_cast<f32>(view::detail::axis_of(d, au)) * 0.5f;
        center_v_ = static_cast<f32>(view::detail::axis_of(d, av)) * 0.5f;
    }

    void mark_dirty() {
        dirty_ = true;
        update();
    }

    // Jump this pane's slice + view to the shared cursor (crosshair navigation).
    void follow_cursor() {
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        (void)au;
        (void)av;
        Vec3f c = st_.cursor;
        slice_ = view::detail::comp_of(c, an);
        clamp_slice_();
        mark_dirty();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        if (dirty_) render_();
        QPainter p(this);
        p.fillRect(rect(), Qt::black);
        if (!qimg_.isNull()) p.drawImage(0, 0, qimg_);
        draw_crosshair_(p);
        draw_annotations_(p);
        p.setPen(QColor(200, 200, 200));
        p.drawText(6, 14, QString("%1  slice %2  lod %3")
                              .arg(axis_ == view::SliceAxis::z ? "xy" : axis_ == view::SliceAxis::y ? "xz" : "yz")
                              .arg(static_cast<double>(slice_), 0, 'f', 1)
                              .arg(img_.lod));
    }

    void resizeEvent(QResizeEvent*) override { mark_dirty(); }

    // VC3D navigation scheme: wheel = zoom at the cursor, shift+wheel = slice stepping
    // (step size = st_.slice_step, Shift+G/H adjusts). Ctrl+wheel kept as plain zoom.
    void wheelEvent(QWheelEvent* e) override {
        const f32 steps = static_cast<f32>(e->angleDelta().y()) / 120.0f;
        if (e->modifiers() & Qt::ShiftModifier) {
            slice_ += steps * static_cast<f32>(st_.slice_step);
            clamp_slice_();
        } else if (e->modifiers() & Qt::ControlModifier) {
            apply_zoom_(std::pow(1.15f, steps));
        } else {
            zoom_at_(std::pow(1.15f, steps), static_cast<f32>(e->position().x()),
                     static_cast<f32>(e->position().y()));
        }
        mark_dirty();
    }

    void enterEvent(QEnterEvent*) override { setFocus(Qt::MouseFocusReason); }

    void keyPressEvent(QKeyEvent* e) override {
        constexpr f32 kPanPx = 64.0f;  // VC3D's arrow-key pan step
        const bool shift = (e->modifiers() & Qt::ShiftModifier) != 0;
        switch (e->key()) {
            case Qt::Key_Left: center_u_ += kPanPx / zoom_; break;
            case Qt::Key_Right: center_u_ -= kPanPx / zoom_; break;
            case Qt::Key_Up: center_v_ += kPanPx / zoom_; break;
            case Qt::Key_Down: center_v_ -= kPanPx / zoom_; break;
            case Qt::Key_Plus:
            case Qt::Key_Equal: apply_zoom_(1.15f); break;
            case Qt::Key_Minus:
            case Qt::Key_Underscore: apply_zoom_(1.0f / 1.15f); break;
            case Qt::Key_M: reset_view_(); break;
            case Qt::Key_R: {  // jump the shared cursor to the voxel under the mouse
                const QPointF mp = mapFromGlobal(QCursor::pos());
                if (!img_.pix.empty() && rect().contains(mp.toPoint())) {
                    st_.cursor = img_.pixel_to_volume(static_cast<f32>(mp.x()), static_cast<f32>(mp.y()));
                    st_.say("focus centered on cursor");
                    st_.refresh();
                }
                return;
            }
            case Qt::Key_X:  // recenter every pane's view on the shared cursor
                st_.say("recentered on focus");
                st_.recenter();
                return;
            case Qt::Key_G:
                if (shift) {
                    st_.slice_step = std::max(1, st_.slice_step - 1);
                    st_.say(std::format("slice step: {}", st_.slice_step));
                }
                return;
            case Qt::Key_H:
                if (shift) {
                    st_.slice_step = std::min(100, st_.slice_step + 1);
                    st_.say(std::format("slice step: {}", st_.slice_step));
                }
                return;
            case Qt::Key_Space:
                st_.show_annotations = !st_.show_annotations;
                st_.say(st_.show_annotations ? "annotations on" : "annotations off");
                st_.refresh();
                return;
            default: e->ignore(); return;  // let the window handle the rest
        }
        mark_dirty();
    }

    void mousePressEvent(QMouseEvent* e) override {
        const f32 px = static_cast<f32>(e->position().x()), py = static_cast<f32>(e->position().y());
        last_ = e->position();
        if (e->button() == Qt::LeftButton) {
            switch (st_.tool) {
                case Tool::navigate: set_cursor_(px, py); break;
                case Tool::stroke: add_stroke_point_(px, py); break;
                case Tool::radial: add_radial_point_(px, py); break;
                case Tool::link: pick_link_(px, py); break;
            }
        }
    }

    void mouseDoubleClickEvent(QMouseEvent* e) override {
        if (e->button() == Qt::LeftButton && (st_.tool == Tool::stroke || st_.tool == Tool::radial)) {
            st_.finish_edit();
            st_.say("edit finished");
            st_.refresh();
        }
    }

    void mouseMoveEvent(QMouseEvent* e) override {
        const QPointF d = e->position() - last_;
        // VC3D mouse scheme: right-drag and middle-drag pan; left-drag pans in navigate mode.
        // Alt+right-drag keeps our window/level (VC3D has no in-viewport W/L binding).
        const bool wl_drag = (e->buttons() & Qt::RightButton) && (e->modifiers() & Qt::AltModifier);
        if (!wl_drag &&
            (e->buttons() & (Qt::MiddleButton | Qt::RightButton) ||
             ((e->buttons() & Qt::LeftButton) && st_.tool == Tool::navigate &&
              (e->modifiers() & Qt::AltModifier) == 0 && d.manhattanLength() > 2))) {
            center_u_ -= static_cast<f32>(d.x()) / zoom_;
            center_v_ -= static_cast<f32>(d.y()) / zoom_;
            last_ = e->position();
            mark_dirty();
        } else if (wl_drag) {  // window/level drag
            const f32 mid = (st_.win_lo + st_.win_hi) * 0.5f + static_cast<f32>(d.x()) * 0.5f;
            const f32 w = std::max(1.0f, (st_.win_hi - st_.win_lo) + static_cast<f32>(d.y()) * 0.5f);
            st_.win_lo = mid - w * 0.5f;
            st_.win_hi = mid + w * 0.5f;
            last_ = e->position();
            st_.refresh();
        } else if (!img_.pix.empty()) {
            const Vec3f p = img_.pixel_to_volume(static_cast<f32>(e->position().x()),
                                                 static_cast<f32>(e->position().y()));
            const s64 ix = std::clamp<s64>(static_cast<s64>(e->position().x()), 0, img_.width - 1);
            const s64 iy = std::clamp<s64>(static_cast<s64>(e->position().y()), 0, img_.height - 1);
            st_.say(std::format("zyx ({:.1f}, {:.1f}, {:.1f})  value {:.0f}", p.z, p.y, p.x,
                                img_.pix[static_cast<usize>(iy * img_.width + ix)]));
        }
    }

public:
    // X / recenter-all: snap this pane's view center (not just its slice) to the shared cursor.
    void recenter_on_cursor() {
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        (void)an;
        Vec3f c = st_.cursor;
        center_u_ = view::detail::comp_of(c, au);
        center_v_ = view::detail::comp_of(c, av);
        follow_cursor();
    }

private:
    void apply_zoom_(f32 f) { zoom_ = std::clamp(zoom_ * f, 1.0f / 64.0f, 64.0f); }

    // Zoom keeping the volume point under the mouse stationary (VC3D zoom-at-cursor).
    void zoom_at_(f32 f, f32 px, f32 py) {
        if (img_.pix.empty()) {
            apply_zoom_(f);
            return;
        }
        Vec3f p = img_.pixel_to_volume(px, py);
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        (void)an;
        apply_zoom_(f);
        center_u_ = view::detail::comp_of(p, au) - (px + 0.5f - static_cast<f32>(width()) * 0.5f) / zoom_;
        center_v_ = view::detail::comp_of(p, av) - (py + 0.5f - static_cast<f32>(height()) * 0.5f) / zoom_;
    }

    // M: fit the whole slice in the pane and recenter (VC3D reset view).
    void reset_view_() {
        const Extent3 d = st_.arch->dims();
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        (void)an;
        const f32 du = static_cast<f32>(view::detail::axis_of(d, au));
        const f32 dv = static_cast<f32>(view::detail::axis_of(d, av));
        zoom_ = std::clamp(std::min(static_cast<f32>(width()) / du, static_cast<f32>(height()) / dv),
                           1.0f / 64.0f, 64.0f);
        center_u_ = du * 0.5f;
        center_v_ = dv * 0.5f;
        st_.say("view reset");
    }

    void clamp_slice_() {
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        (void)au;
        (void)av;
        slice_ = std::clamp(slice_, 0.0f, static_cast<f32>(view::detail::axis_of(st_.arch->dims(), an) - 1));
    }

    void render_() {
        dirty_ = false;
        view::SliceSpec sp;
        sp.axis = axis_;
        sp.slice = slice_;
        sp.center_u = center_u_;
        sp.center_v = center_v_;
        sp.zoom = zoom_;
        sp.width = std::max(1, width());
        sp.height = std::max(1, height());
        auto r = st_.engine->render(sp);
        if (!r) {
            st_.say("render failed: " + r.error().message);
            return;
        }
        img_ = std::move(*r);
        const std::vector<u8> gray = img_.to_u8(st_.win_lo, st_.win_hi);
        qimg_ = QImage(static_cast<int>(img_.width), static_cast<int>(img_.height), QImage::Format_Grayscale8);
        for (s64 y = 0; y < img_.height; ++y)
            std::memcpy(qimg_.scanLine(static_cast<int>(y)), gray.data() + y * img_.width,
                        static_cast<usize>(img_.width));
        st_.engine->prefetch_around(sp);
    }

    void set_cursor_(f32 px, f32 py) {
        if (img_.pix.empty()) return;
        st_.cursor = img_.pixel_to_volume(px, py);
        st_.refresh();
    }

    [[nodiscard]] Vec3f pick_point_(f32 px, f32 py) const { return img_.pixel_to_volume(px, py); }

    // Snap to the brightest pixel in a small disc (sheets are bright in CT) — the assisted
    // part of the radial tool: a rough click lands ON the wrap crossing.
    [[nodiscard]] Vec3f pick_ridge_(f32 px, f32 py) const {
        if (img_.pix.empty()) return pick_point_(px, py);
        constexpr int R = 6;
        f32 best = -1e30f;
        f32 bx = px, by = py;
        for (int dy = -R; dy <= R; ++dy)
            for (int dx = -R; dx <= R; ++dx) {
                if (dx * dx + dy * dy > R * R) continue;
                const s64 x = static_cast<s64>(px) + dx, y = static_cast<s64>(py) + dy;
                if (x < 0 || y < 0 || x >= img_.width || y >= img_.height) continue;
                const f32 v = img_.pix[static_cast<usize>(y * img_.width + x)];
                if (v > best) {
                    best = v;
                    bx = static_cast<f32>(x);
                    by = static_cast<f32>(y);
                }
            }
        return pick_point_(bx, by);
    }

    void add_stroke_point_(f32 px, f32 py) {
        if (st_.active_stroke < 0) {
            annotate::CoWindingStroke s;
            s.kind = st_.stroke_kind;
            s.name = std::format("stroke-{}", st_.anno.strokes.size());
            st_.anno.strokes.push_back(std::move(s));
            st_.active_stroke = static_cast<s32>(st_.anno.strokes.size()) - 1;
        }
        st_.anno.strokes[static_cast<usize>(st_.active_stroke)].points.push_back(pick_point_(px, py));
        st_.refresh();
    }

    void add_radial_point_(f32 px, f32 py) {
        if (st_.active_radial < 0) {
            annotate::RadialLine r;
            r.name = std::format("radial-{}", st_.anno.radial_lines.size());
            st_.anno.radial_lines.push_back(std::move(r));
            st_.active_radial = static_cast<s32>(st_.anno.radial_lines.size()) - 1;
        }
        annotate::RadialLine& r = st_.anno.radial_lines[static_cast<usize>(st_.active_radial)];
        r.points.push_back(r.points.empty() ? pick_point_(px, py) : pick_ridge_(px, py));
        r.offset.push_back(r.offset.empty() ? 0 : r.offset.back() + 1);
        st_.say(std::format("radial +{}", r.offset.back()));
        st_.refresh();
    }

    void pick_link_(f32 px, f32 py) {
        // Nearest stroke by any point within 12 px.
        const Vec3f q = pick_point_(px, py);
        s32 best = -1;
        f32 bestd = 12.0f / zoom_;
        for (usize i = 0; i < st_.anno.strokes.size(); ++i)
            for (const Vec3f& pt : st_.anno.strokes[i].points) {
                const Vec3f d = pt - q;
                const f32 dist = norm(d);
                if (dist < bestd) {
                    bestd = dist;
                    best = static_cast<s32>(i);
                }
            }
        if (best < 0) return;
        if (st_.link_a < 0) {
            st_.link_a = best;
            st_.say(std::format("link: stroke {} picked (shift-click second = cannot-link)", best));
        } else if (best != st_.link_a) {
            const bool cannot = (QGuiApplication::keyboardModifiers() & Qt::ShiftModifier) != 0;
            st_.anno.links.push_back({st_.link_a, best, cannot});
            st_.say(std::format("{}-link {} <-> {}", cannot ? "cannot" : "must", st_.link_a, best));
            st_.link_a = -1;
        }
        st_.refresh();
    }

    // ---- overlay drawing ----
    // A point is on this pane if its normal-axis coord is within the slab tolerance.
    [[nodiscard]] bool on_slice_(Vec3f p, f32& px, f32& py) const {
        int au, av, an;
        view::detail::plane_axes(axis_, au, av, an);
        (void)au;
        (void)av;
        Vec3f q = p;
        if (std::abs(view::detail::comp_of(q, an) - slice_) > std::max(1.5f, 1.0f / zoom_)) return false;
        img_.volume_to_pixel(p, px, py);
        return px >= -50 && py >= -50 && px < static_cast<f32>(width()) + 50 && py < static_cast<f32>(height()) + 50;
    }

    void draw_crosshair_(QPainter& p) const {
        if (img_.pix.empty()) return;
        f32 px = 0, py = 0;
        img_.volume_to_pixel(st_.cursor, px, py);
        p.setPen(QColor(0, 255, 255, 140));
        p.drawLine(QPointF(px, 0), QPointF(px, height()));
        p.drawLine(QPointF(0, py), QPointF(width(), py));
    }

    void draw_annotations_(QPainter& p) const {
        if (img_.pix.empty() || !st_.show_annotations) return;
        p.setRenderHint(QPainter::Antialiasing);
        for (usize i = 0; i < st_.anno.strokes.size(); ++i) {
            const auto& s = st_.anno.strokes[i];
            QColor c = detail::stroke_color(s.kind);
            if (static_cast<s32>(i) == st_.active_stroke || static_cast<s32>(i) == st_.link_a) c = c.lighter(150);
            p.setPen(QPen(c, 2));
            QPointF prev;
            bool has_prev = false;
            for (const Vec3f& pt : s.points) {
                f32 px, py;
                if (!on_slice_(pt, px, py)) { has_prev = false; continue; }
                p.drawEllipse(QPointF(px, py), 2.5, 2.5);
                if (has_prev) p.drawLine(prev, QPointF(px, py));
                prev = QPointF(px, py);
                has_prev = true;
            }
            if (s.has_winding && !s.points.empty()) {
                f32 px, py;
                if (on_slice_(s.points.front(), px, py))
                    p.drawText(QPointF(px + 5, py - 5), QString("W=%1").arg(static_cast<double>(s.winding)));
            }
        }
        for (usize i = 0; i < st_.anno.radial_lines.size(); ++i) {
            const auto& r = st_.anno.radial_lines[i];
            QColor c(255, 60, 60);
            if (static_cast<s32>(i) == st_.active_radial) c = c.lighter(150);
            p.setPen(QPen(c, 2));
            QPointF prev;
            bool has_prev = false;
            for (usize k = 0; k < r.points.size(); ++k) {
                f32 px, py;
                if (!on_slice_(r.points[k], px, py)) { has_prev = false; continue; }
                p.drawEllipse(QPointF(px, py), 3, 3);
                p.drawText(QPointF(px + 5, py - 5), QString("+%1").arg(r.offset[k]));
                if (has_prev) p.drawLine(prev, QPointF(px, py));
                prev = QPointF(px, py);
                has_prev = true;
            }
        }
    }

    ViewerState& st_;
    view::SliceAxis axis_;
    f32 slice_ = 0, center_u_ = 0, center_v_ = 0, zoom_ = 1.0f;
    bool dirty_ = true;
    view::SliceImage img_;
    QImage qimg_;
    QPointF last_;
};

class SurfacePane : public QWidget {
public:
    explicit SurfacePane(ViewerState& st, QWidget* parent = nullptr) : QWidget(parent), st_(st) {
        setMinimumSize(160, 160);
    }
    void mark_dirty() {
        dirty_ = true;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.fillRect(rect(), Qt::black);
        if (!st_.has_surface) {
            p.setPen(QColor(150, 150, 150));
            p.drawText(rect(), Qt::AlignCenter, "no surface loaded (--surf s.fxsurf)");
            return;
        }
        if (dirty_) render_();
        if (!qimg_.isNull())
            p.drawImage(rect(), qimg_);  // scaled blit; uv grid keeps its aspect via the widget layout
        p.setPen(QColor(200, 200, 200));
        p.drawText(6, 14, QString("surface  mode %1").arg(static_cast<int>(st_.comp_mode)));
    }

    void mousePressEvent(QMouseEvent* e) override {
        if (!st_.has_surface || e->button() != Qt::LeftButton || qimg_.isNull()) return;
        // widget px -> uv cell -> 3D cursor (crosshair through the surface).
        const s64 u = std::clamp<s64>(static_cast<s64>(e->position().x() / width() * st_.surface.nu), 0,
                                      st_.surface.nu - 1);
        const s64 v = std::clamp<s64>(static_cast<s64>(e->position().y() / height() * st_.surface.nv), 0,
                                      st_.surface.nv - 1);
        if (!st_.surface.is_valid(u, v)) return;
        st_.cursor = st_.surface.at(u, v);
        st_.refresh();
    }

private:
    void render_() {
        dirty_ = false;
        view::CompositeSpec cs;
        cs.mode = st_.comp_mode;
        cs.lo = st_.comp_lo;
        cs.hi = st_.comp_hi;
        auto r = view::render_surface_composite(*st_.arch, st_.surface, cs);
        if (!r) {
            st_.say("surface render failed: " + r.error().message);
            return;
        }
        // Window the composite with the shared W/L (alpha/beer come back in value units too).
        const f32 lo = st_.win_lo, hi = std::max(st_.win_lo + 1.0f, st_.win_hi);
        qimg_ = QImage(static_cast<int>(r->width), static_cast<int>(r->height), QImage::Format_Grayscale8);
        for (s64 y = 0; y < r->height; ++y) {
            u8* line = qimg_.scanLine(static_cast<int>(y));
            for (s64 x = 0; x < r->width; ++x) {
                const usize i = static_cast<usize>(y * r->width + x);
                line[x] = r->valid[i]
                              ? static_cast<u8>(std::clamp((r->pix[i] - lo) / (hi - lo) * 255.0f, 0.0f, 255.0f))
                              : 0;
            }
        }
    }

    ViewerState& st_;
    bool dirty_ = true;
    QImage qimg_;
};

}  // namespace fenix::gui
