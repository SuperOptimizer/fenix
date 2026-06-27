// core/test.hpp — first-party test harness (STUB but usable).
// TEST(name){...} auto-registers; REQUIRE (fatal) / CHECK (non-fatal). Define
// FENIX_TEST_MAIN in exactly one TU (each test is its own per-file binary) to get main().
#pragma once

#include <print>
#include <string_view>
#include <vector>

namespace fenix::test {

struct Case {
    std::string_view name;
    void (*fn)();
};

inline std::vector<Case>& cases() {
    static std::vector<Case> c;
    return c;
}

inline int add(Case c) {
    cases().push_back(c);
    return 0;
}

inline int& failures() {
    static int n = 0;
    return n;
}

inline int run_all() {
    std::size_t failed = 0;
    for (const auto& c : cases()) {
        int before = failures();
        c.fn();
        bool ok = failures() == before;
        std::println("[{}] {}", ok ? "PASS" : "FAIL", c.name);
        failed += ok ? 0u : 1u;
    }
    std::println("{}/{} cases passed", cases().size() - failed, cases().size());
    return failed == 0 ? 0 : 1;
}

}  // namespace fenix::test

#define TEST(NAME)                                                                       \
    static void fenix_test_##NAME();                                                     \
    namespace {                                                                          \
    [[maybe_unused]] const int fenix_test_reg_##NAME =                                   \
        ::fenix::test::add({#NAME, &fenix_test_##NAME});                                 \
    }                                                                                    \
    static void fenix_test_##NAME()

// Variadic so braced-initializer commas (e.g. CHECK(a == Extent3{1,2,3})) are accepted.
// Compare with tolerances, never bit-exact (fast-math). Use REQUIRE for invariants.
#define CHECK(...)                                                                       \
    do {                                                                                 \
        if (!(__VA_ARGS__)) {                                                            \
            ::fenix::test::failures()++;                                                 \
            std::println(stderr, "  CHECK failed: {} ({}:{})", #__VA_ARGS__, __FILE__,   \
                         __LINE__);                                                      \
        }                                                                                \
    } while (0)

#define REQUIRE(...)                                                                     \
    do {                                                                                 \
        if (!(__VA_ARGS__)) {                                                            \
            ::fenix::test::failures()++;                                                 \
            std::println(stderr, "  REQUIRE failed: {} ({}:{})", #__VA_ARGS__, __FILE__, \
                         __LINE__);                                                      \
            return;                                                                      \
        }                                                                                \
    } while (0)

#ifdef FENIX_TEST_MAIN
int main() { return ::fenix::test::run_all(); }
#endif
