// annotate/annotation.hpp — the constraint annotations the tracer + spiral fit consume:
// co-winding strokes (anything contiguous on ONE winding: patch extracts, fibers, a
// kollesis, hand drawings), radial winding lines (+1/+2/+3 crossings walking outward),
// sparse normal hints, and must/cannot links between strokes. Versioned TOML on disk
// (core::Config reader + first-party writer, atomic temp-rename, unknown version
// rejected). The umbilicus stays its own type (umbilicus.hpp). See annotate/CLAUDE.md,
// docs/design/viewer-annotation.md.
#pragma once

#include "core/config.hpp"
#include "core/core.hpp"

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

namespace fenix::annotate {

inline constexpr s64 kAnnotationVersion = 1;

enum class StrokeKind : u8 { generic, patch_extract, trace_extract, fiber, kollesis, drawing };

// A contiguous set of points known to lie on ONE (possibly unknown) winding. The winding
// scalar is constant along a sheet arm, so a stroke of ANY size/shape on one sheet is a
// valid co-winding constraint — a small trusted patch, a fiber, a full-height kollesis.
struct CoWindingStroke {
    std::string name;
    StrokeKind kind = StrokeKind::generic;
    std::vector<Vec3f> points;  // ZYX voxel coords
    f32 weight = 1.0f;
    bool has_winding = false;  // absolute winding known (calibrates the global scale)
    f32 winding = 0.0f;
};

// Ordered wrap crossings walking radially outward: points[i] lies offset[i] windings from
// points[0] (typically 0,1,2,3,...). With a known base winding every point becomes an
// absolute constraint; without one, consecutive pairs become relative-winding constraints.
struct RadialLine {
    std::string name;
    std::vector<Vec3f> points;
    std::vector<s32> offset;  // one per point, relative to points[0]
    f32 weight = 1.0f;
    bool has_base_winding = false;
    f32 base_winding = 0.0f;
};

// Sparse across-sheet normal at a point (the "lasagna" direction term, consumed when the
// dense normals loss lands — winding roadmap P4).
struct NormalHint {
    Vec3f pos;
    Vec3f dir;
    f32 weight = 1.0f;
};

// must-link (same winding — the bridge merges the two strokes' co-winding groups) or
// cannot-link (different windings — a hard repulsive edge for segmentation).
struct StrokeLink {
    s32 a = -1, b = -1;  // indices into AnnotationSet::strokes
    bool cannot = false;
};

struct AnnotationSet {
    std::vector<CoWindingStroke> strokes;
    std::vector<RadialLine> radial_lines;
    std::vector<NormalHint> normals;
    std::vector<StrokeLink> links;

    [[nodiscard]] bool empty() const {
        return strokes.empty() && radial_lines.empty() && normals.empty() && links.empty();
    }
};

namespace detail {

inline const char* kind_name(StrokeKind k) {
    switch (k) {
        case StrokeKind::patch_extract: return "patch";
        case StrokeKind::trace_extract: return "trace";
        case StrokeKind::fiber: return "fiber";
        case StrokeKind::kollesis: return "kollesis";
        case StrokeKind::drawing: return "drawing";
        default: return "generic";
    }
}

inline Expected<StrokeKind> kind_from(const std::string& s) {
    if (s == "generic") return StrokeKind::generic;
    if (s == "patch") return StrokeKind::patch_extract;
    if (s == "trace") return StrokeKind::trace_extract;
    if (s == "fiber") return StrokeKind::fiber;
    if (s == "kollesis") return StrokeKind::kollesis;
    if (s == "drawing") return StrokeKind::drawing;
    return err(Errc::decode_error, "unknown stroke kind: " + s);
}

inline void append_vec3s(std::string& out, const char* key, std::span<const Vec3f> pts) {
    out += key;
    out += " = [";
    for (usize i = 0; i < pts.size(); ++i) {
        if (i) out += ", ";
        out += std::format("{}, {}, {}", pts[i].z, pts[i].y, pts[i].x);
    }
    out += "]\n";
}

inline Expected<std::vector<Vec3f>> parse_vec3s(const Config& c, const std::string& key) {
    const std::vector<std::string> raw = c.get_array(key);
    if (raw.size() % 3 != 0) return err(Errc::decode_error, key + ": length not a multiple of 3");
    std::vector<Vec3f> out(raw.size() / 3);
    for (usize i = 0; i < out.size(); ++i) {
        f32 v[3];
        for (int k = 0; k < 3; ++k) {
            const std::string& s = raw[3 * i + static_cast<usize>(k)];
            f32 x = 0;
            const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), x);
            if (ec != std::errc{}) return err(Errc::decode_error, key + ": bad float " + s);
            v[k] = x;
        }
        out[i] = Vec3f{v[0], v[1], v[2]};
    }
    return out;
}

}  // namespace detail

inline Expected<void> save_annotations(const AnnotationSet& a, const std::string& path) {
    std::string out;
    out += std::format("version = {}\n", kAnnotationVersion);
    for (usize i = 0; i < a.strokes.size(); ++i) {
        const CoWindingStroke& s = a.strokes[i];
        out += std::format("\n[stroke.{}]\n", i);
        out += std::format("name = \"{}\"\n", s.name);
        out += std::format("kind = \"{}\"\n", detail::kind_name(s.kind));
        out += std::format("weight = {}\n", s.weight);
        if (s.has_winding) out += std::format("winding = {}\n", s.winding);
        detail::append_vec3s(out, "points", s.points);
    }
    for (usize i = 0; i < a.radial_lines.size(); ++i) {
        const RadialLine& r = a.radial_lines[i];
        out += std::format("\n[radial.{}]\n", i);
        out += std::format("name = \"{}\"\n", r.name);
        out += std::format("weight = {}\n", r.weight);
        if (r.has_base_winding) out += std::format("base_winding = {}\n", r.base_winding);
        detail::append_vec3s(out, "points", r.points);
        out += "offsets = [";
        for (usize k = 0; k < r.offset.size(); ++k) {
            if (k) out += ", ";
            out += std::format("{}", r.offset[k]);
        }
        out += "]\n";
    }
    for (usize i = 0; i < a.normals.size(); ++i) {
        const NormalHint& n = a.normals[i];
        out += std::format("\n[normal.{}]\n", i);
        out += std::format("pos = [{}, {}, {}]\n", n.pos.z, n.pos.y, n.pos.x);
        out += std::format("dir = [{}, {}, {}]\n", n.dir.z, n.dir.y, n.dir.x);
        out += std::format("weight = {}\n", n.weight);
    }
    for (usize i = 0; i < a.links.size(); ++i) {
        const StrokeLink& l = a.links[i];
        out += std::format("\n[link.{}]\na = {}\nb = {}\ncannot = {}\n", i, l.a, l.b, l.cannot);
    }

    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return err(Errc::io_error, "cannot open " + tmp);
    f.write(out.data(), static_cast<std::streamsize>(out.size()));
    f.close();
    if (!f) return err(Errc::io_error, "write failed: " + tmp);
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) return err(Errc::io_error, "rename failed: " + path);
    return {};
}

inline Expected<AnnotationSet> load_annotations(const std::string& path) {
    auto cfg = Config::load(path);
    if (!cfg) return std::unexpected(cfg.error());
    const Config& c = *cfg;
    const auto ver = c.get_int("version");
    if (!ver || *ver != kAnnotationVersion)
        return err(Errc::unsupported, std::format("annotation version {} (want {}): {}",
                                                  ver.value_or(-1), kAnnotationVersion, path));
    AnnotationSet a;
    for (s64 i = 0; c.has(std::format("stroke.{}.points", i)); ++i) {
        const std::string sec = std::format("stroke.{}.", i);
        CoWindingStroke s;
        s.name = c.get_string(sec + "name").value_or("");
        auto kind = detail::kind_from(c.get_string(sec + "kind").value_or("generic"));
        if (!kind) return std::unexpected(kind.error());
        s.kind = *kind;
        s.weight = static_cast<f32>(c.get_float(sec + "weight").value_or(1.0));
        if (const auto w = c.get_float(sec + "winding")) {
            s.has_winding = true;
            s.winding = static_cast<f32>(*w);
        }
        auto pts = detail::parse_vec3s(c, sec + "points");
        if (!pts) return std::unexpected(pts.error());
        s.points = std::move(*pts);
        a.strokes.push_back(std::move(s));
    }
    for (s64 i = 0; c.has(std::format("radial.{}.points", i)); ++i) {
        const std::string sec = std::format("radial.{}.", i);
        RadialLine r;
        r.name = c.get_string(sec + "name").value_or("");
        r.weight = static_cast<f32>(c.get_float(sec + "weight").value_or(1.0));
        if (const auto w = c.get_float(sec + "base_winding")) {
            r.has_base_winding = true;
            r.base_winding = static_cast<f32>(*w);
        }
        auto pts = detail::parse_vec3s(c, sec + "points");
        if (!pts) return std::unexpected(pts.error());
        r.points = std::move(*pts);
        for (const std::string& s : c.get_array(sec + "offsets")) {
            s32 v = 0;
            const auto [p, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
            if (ec != std::errc{}) return err(Errc::decode_error, sec + "offsets: bad int " + s);
            r.offset.push_back(v);
        }
        if (r.offset.size() != r.points.size())
            return err(Errc::decode_error, sec + "offsets/points size mismatch");
        a.radial_lines.push_back(std::move(r));
    }
    for (s64 i = 0; c.has(std::format("normal.{}.pos", i)); ++i) {
        const std::string sec = std::format("normal.{}.", i);
        NormalHint n;
        auto pos = detail::parse_vec3s(c, sec + "pos");
        auto dir = detail::parse_vec3s(c, sec + "dir");
        if (!pos) return std::unexpected(pos.error());
        if (!dir) return std::unexpected(dir.error());
        if (pos->size() != 1 || dir->size() != 1)
            return err(Errc::decode_error, sec + "pos/dir must be one triple");
        n.pos = (*pos)[0];
        n.dir = (*dir)[0];
        n.weight = static_cast<f32>(c.get_float(sec + "weight").value_or(1.0));
        a.normals.push_back(n);
    }
    for (s64 i = 0; c.has(std::format("link.{}.a", i)); ++i) {
        const std::string sec = std::format("link.{}.", i);
        StrokeLink l;
        l.a = static_cast<s32>(c.get_int(sec + "a").value_or(-1));
        l.b = static_cast<s32>(c.get_int(sec + "b").value_or(-1));
        l.cannot = c.get_bool(sec + "cannot").value_or(false);
        const s32 ns = static_cast<s32>(a.strokes.size());
        if (l.a < 0 || l.b < 0 || l.a >= ns || l.b >= ns)
            return err(Errc::decode_error, sec + "stroke index out of range");
        a.links.push_back(l);
    }
    return a;
}

}  // namespace fenix::annotate
