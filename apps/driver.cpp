// driver.cpp — the fenix entry point. Two build modes (see ADR / CMake FENIX_SPLIT):
//   * UNITY (default): include the umbrella header -> the whole program is ONE translation unit; every
//     module self-registers its stage(s) transitively. Fewest total CPU-seconds; best for clean/CI.
//   * SPLIT (-DFENIX_SPLIT): each module is compiled as its OWN TU (src/units/<mod>.cpp) in parallel and
//     linked in; the module .o's static registrars populate the registry at startup. driver then includes
//     ONLY what main() itself touches (core + config + the archive for `info`) so it stays a light TU and
//     does NOT re-register every stage. Enables parallel + incremental dev builds.
// It dispatches argv to the stage registry and must stay tiny: parse -> Context -> registry -> run.
// Adding a stage NEVER edits this file.
#if defined(FENIX_SPLIT)
#include "codec/archive.hpp"
#include "core/config.hpp"
#include "core/core.hpp"
#else
#include "fenix.hpp"
#endif

#include <print>
#include <string_view>
#include <vector>

namespace {

void print_help() {
    std::println("fenix {} — virtual scroll unrolling", fenix::version);
    std::println("usage: fenix <subcommand> [options]");
    std::println("\nsubcommands:");
    std::println("  {:<12} {}", "help", "show this help");
    std::println("  {:<12} {}", "info", "inspect an artifact (--json for tooling)");
    std::println("  {:<12} {}", "run", "execute a .fxrecipe pipeline (sequential stages)");
    for (const auto& s : fenix::registry())
        std::println("  {:<12} {}", s.name, s.help);
}

}  // namespace

int main(int argc, char** argv) {
    // Clamp OpenMP's default team size to the real CPU budget (cgroup quota, not the host core count) BEFORE
    // any parallel region — in a container nproc reports the host's cores, so libomp would otherwise spawn
    // ~256 threads onto a ~27-CPU quota and thrash. Override with FENIX_THREADS / OMP_NUM_THREADS.
    fenix::init_thread_limits();

    std::vector<std::string_view> raw(argv + (argc > 0 ? 1 : 0), argv + argc);

    // Verbosity flags anywhere on the line set the global log level (default info, or $FENIX_LOG_LEVEL):
    // -v -> debug, -vv/-vvv/--trace -> trace, --quiet -> error. They are stripped before dispatch.
    std::vector<std::string_view> args;
    for (std::string_view a : raw) {
        if (a == "-v") fenix::log_level() = fenix::LogLevel::debug;
        else if (a == "-vv" || a == "-vvv" || a == "--trace") fenix::log_level() = fenix::LogLevel::trace;
        else if (a == "--quiet") fenix::log_level() = fenix::LogLevel::error;
        else args.push_back(a);
    }

    if (args.empty() || args[0] == "help" || args[0] == "-h" || args[0] == "--help") {
        print_help();
        return 0;
    }

    fenix::Context ctx;
    const std::string_view cmd = args[0];
    const std::span<const std::string_view> rest(args.begin() + 1, args.end());

    if (cmd == "info") {
        if (rest.empty()) {
            std::println(stderr, "usage: fenix info <artifact.fxvol>");
            return 2;
        }
        auto a = fenix::codec::VolumeArchive::open(std::string(rest[0]));
        if (!a) {
            std::println(stderr, "info: {}", a.error().message);
            return 1;
        }
        const auto d = a->dims();
        const auto ce = a->chunk_extent();
        const auto bp = a->params();
        fenix::s64 real = 0, zero = 0, absent = 0;
        for (fenix::s64 cz = 0; cz < ce.z; ++cz)
            for (fenix::s64 cy = 0; cy < ce.y; ++cy)
                for (fenix::s64 cx = 0; cx < ce.x; ++cx)
                    switch (a->coverage({cz, cy, cx})) {
                        case fenix::codec::Coverage::Real: ++real; break;
                        case fenix::codec::Coverage::Zero: ++zero; break;
                        case fenix::codec::Coverage::Absent: ++absent; break;
                    }
        std::println("fxvol  dims(ZYX)={}x{}x{}  chunks={}x{}x{}  codec=dct q={} hf_exp={} dz_frac={}",
                     d.z, d.y, d.x, ce.z, ce.y, ce.x, bp.q, bp.hf_exp, bp.dz_frac);
        std::println("coverage  real={}  zero={}  absent={}  ({} total)", real, zero, absent,
                     ce.z * ce.y * ce.x);
        return 0;
    }

    if (cmd == "run") {
        if (rest.empty()) {
            std::println(stderr, "usage: fenix run <recipe.fxrecipe>");
            return 2;
        }
        auto cfg = fenix::Config::load(std::string(rest[0]));
        if (!cfg) {
            std::println(stderr, "run: {}", cfg.error().message);
            return 1;
        }
        const auto stages = cfg->get_array("stages");
        if (stages.empty()) {
            std::println(stderr, "run: recipe has no 'stages' array");
            return 1;
        }
        fenix::Context ctx2;
        for (const std::string& sname : stages) {
            const fenix::Stage* st = fenix::find_stage(sname);
            if (!st) {
                std::println(stderr, "run: unknown stage '{}'", sname);
                return 2;
            }
            // Per-stage args from "<stage>.args" array (stored as owned strings).
            const std::vector<std::string> sargs = cfg->get_array(sname + ".args");
            std::vector<std::string_view> sviews(sargs.begin(), sargs.end());
            fenix::log(fenix::LogLevel::info, "run: stage '{}'", sname);
            auto res = st->run(sviews, ctx2);
            if (!res) {
                std::println(stderr, "run: stage '{}' failed: {}", sname, res.error().message);
                return 1;
            }
        }
        fenix::log(fenix::LogLevel::info, "run: recipe complete ({} stages)", stages.size());
        return 0;
    }

    const fenix::Stage* stage = fenix::find_stage(cmd);
    if (!stage) {
        std::println(stderr, "unknown subcommand: {}", cmd);
        print_help();
        return 2;
    }

    fenix::Expected<int> result = stage->run(rest, ctx);
    if (!result) {
        std::println(stderr, "error: {}", result.error().message);
        return 1;
    }
    return *result;
}
