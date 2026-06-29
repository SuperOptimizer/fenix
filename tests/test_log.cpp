// test_log.cpp — the logging system (core/log.hpp): level gating, module tag + format, thread-safe
// whole-record emission (no torn lines from parallel_for), and the scoped timer. Uses a capturing sink
// so the assertions don't depend on stderr.
#define FENIX_TEST_MAIN
#include "core/core.hpp"
#include "core/test.hpp"

#include <string>
#include <vector>

using namespace fenix;

namespace {
// Install a capturing sink; returns the captured records and restores the previous sink + level.
struct Capture {
    std::vector<std::string> recs;
    LogSink prev;
    LogLevel prev_lvl;
    explicit Capture(LogLevel lvl) : prev(log_sink()), prev_lvl(log_level()) {
        log_level() = lvl;
        log_sink() = [this](std::string_view r) { recs.emplace_back(r); };
    }
    ~Capture() {
        log_sink() = prev;
        log_level() = prev_lvl;
    }
    bool any_contains(std::string_view s) const {
        for (const std::string& r : recs)
            if (r.find(s) != std::string::npos) return true;
        return false;
    }
};
}  // namespace

TEST(log_level_gating) {
    Capture cap(LogLevel::warn);
    FENIX_INFO("t", "info should be suppressed");
    FENIX_DEBUG("t", "debug should be suppressed");
    FENIX_WARN("t", "warn shows {}", 1);
    FENIX_ERROR("t", "error shows");
    CHECK(cap.recs.size() == 2);
    CHECK(!cap.any_contains("suppressed"));
    CHECK(cap.any_contains("warn shows 1"));
}

TEST(log_module_tag_and_format) {
    Capture cap(LogLevel::trace);
    FENIX_ERROR("winding", "loss {:.2f} -> {}", 0.5, 7);
    REQUIRE(cap.recs.size() == 1);
    CHECK(cap.recs[0].find("[E]") != std::string::npos);
    CHECK(cap.recs[0].find("[winding]") != std::string::npos);
    CHECK(cap.recs[0].find("loss 0.50 -> 7") != std::string::npos);
    CHECK(cap.recs[0].back() == '\n');  // newline-terminated whole record
}

TEST(log_threadsafe_whole_records) {
    Capture cap(LogLevel::info);
    const s64 n = 2000;
    parallel_for(0, n, [&](s64 i) { FENIX_INFO("par", "row {}", i); });
    CHECK(static_cast<s64>(cap.recs.size()) == n);  // every record arrived
    bool ok = true;
    for (const std::string& r : cap.recs)
        if (r.empty() || r.front() != '[' || r.back() != '\n' || r.find("[par]") == std::string::npos) ok = false;
    CHECK(ok);  // no torn / interleaved lines (sink gets one complete record at a time)
}

TEST(log_scope_timer_fires) {
    Capture cap(LogLevel::debug);
    { FENIX_SCOPE_TIMER("seg", "work"); }
    CHECK(cap.any_contains("work:"));
    CHECK(cap.any_contains(" s"));
}
